# -*- coding: utf-8 -*-
import json
import logging
import os
import io
import re
import hashlib
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader

# 尝试导入 redis
try:
    import redis

    HAS_REDIS_LIB = True
except ImportError:
    HAS_REDIS_LIB = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

REQUEST_ID_HEADER = 'x-fc-request-id'

# 初始化 Flask
app = Flask(__name__)
CORS(app)

# ===========================
#   环境变量配置
# ===========================
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen-plus")

# Redis 配置
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)


# ===========================
#   缓存管理类
# ===========================
class CacheManager:
    def __init__(self):
        self.redis_client = None
        self.local_cache = {}
        self.use_redis = False

        if HAS_REDIS_LIB and os.getenv("REDIS_HOST"):
            try:
                self.redis_client = redis.Redis(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    password=REDIS_PASSWORD,
                    decode_responses=True,
                    socket_connect_timeout=1
                )
                self.redis_client.ping()
                self.use_redis = True
                logger.info("✅ Redis 连接成功")
            except Exception as e:
                logger.warning(f"⚠️ Redis 连接失败: {e}，降级为内存缓存")
        else:
            logger.info("ℹ️ 使用内存缓存")

    def generate_key(self, file_bytes, jd_text):
        # 计算 MD5 作为唯一指纹
        content_hash = hashlib.md5(file_bytes).hexdigest()
        jd_hash = hashlib.md5(jd_text.encode('utf-8')).hexdigest()
        return f"resume:v3:{content_hash}:{jd_hash}"

    def get(self, key):
        if self.use_redis:
            try:
                data = self.redis_client.get(key)
                return json.loads(data) if data else None
            except:
                return None
        return self.local_cache.get(key)

    def set(self, key, data, expire=3600):
        if self.use_redis:
            try:
                self.redis_client.setex(key, expire, json.dumps(data, ensure_ascii=False))
            except Exception as e:
                logger.error(f"Redis Set Error: {e}")
        else:
            self.local_cache[key] = data


cache_manager = CacheManager()


# ===========================
#   核心逻辑
# ===========================
def extract_text_from_pdf(file_bytes):
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        texts = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(texts)
    except Exception as e:
        logger.error(f"PDF Error: {e}")
        return ""


def call_qwen_analysis(resume_text, jd_text):
    if not QWEN_API_KEY:
        raise Exception("未配置 QWEN_API_KEY")

    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json",
    }

    # 🔥 核心升级：精准评分 Prompt (Prompt Engineering) 🔥
    # 满足要求：
    # 1. 关键词提取 (Keywords)
    # 2. 精准评分 (基于权重计算)
    # 3. 结构化对比
    prompt = f"""
    你是一名资深技术面试官和简历分析专家。请对比【候选人简历】与【岗位JD】，进行深度匹配分析。

    【岗位JD】：
    {jd_text[:1000]}

    【候选人简历】：
    {resume_text[:3000]}

    请执行以下步骤进行分析（思维链）：
    1. 【关键词提取】：从JD中提取3-5个核心技术关键词（Keywords）。
    2. 【信息提取】：从简历中提取基本信息。
    3. 【精准评分】：请严格按照以下权重进行打分，并计算总分（0-100）：
       - 技能匹配度 (权重40%): 核心关键词的覆盖率和熟练度。
       - 经验匹配度 (权重30%): 工作年限、项目复杂度与JD的契合度。
       - 学历与基础 (权重20%): 学历背景、专业是否达标。
       - 综合素质 (权重10%): 稳定性、沟通描述等。

    请输出严格的 JSON 格式（不要包含 Markdown 代码块）：
    {{
      "basic_info": {{ 
        "name": "姓名", 
        "email": "邮箱", 
        "phone": "电话", 
        "address": "居住地址(未提及则填'未提及')",
        "education": "最高学历(如: 本科, 硕士)",
        "years_of_experience": "工作年限(如: 3年, 应届生，在校生)",
        "job_intention": "求职意向"
      }},
      "jd_analysis": {{
        "keywords": ["关键词1", "关键词2", "关键词3"]
      }},
      "education_background": [ "学历详细背景1 (时间-学校-专业)", "学历详细背景2"],
      "match_score": 0,
      "score_breakdown": {{
         "skill_score": 0,
         "experience_score": 0,
         "education_score": 0,
         "general_score": 0
      }},
      "summary": "候选人画像总结(100字内)",
      "match_analysis": "详细分析报告：\\n1. 核心优势：...\\n2. 差距分析：...\\n3. 综合建议：..."
    }}
    """

    body = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1  # 低温度保证评分严谨
    }

    try:
        res = requests.post(url, headers=headers, json=body, timeout=60)
        return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        raise


# ===========================
#   路由接口
# ===========================
@app.route('/check/analyze', methods=['POST'])
def analyze():
    rid = request.headers.get(REQUEST_ID_HEADER, "")
    logger.info(f"Start Request: {rid}")

    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        jd = request.form.get('jd', '')

        # 读取文件
        file_bytes = file.read()
        if not file_bytes: return jsonify({"error": "Empty file"}), 400

        # 1. 检查缓存 (Cache Check)
        cache_key = cache_manager.generate_key(file_bytes, jd)
        cached_data = cache_manager.get(cache_key)

        if cached_data:
            logger.info("⚡️ Cache Hit")
            cached_data["_is_cached"] = True
            return jsonify(cached_data)

        # 2. PDF 解析
        resume_text = extract_text_from_pdf(file_bytes)
        if not resume_text.strip():
            return jsonify({"error": "PDF解析为空，请检查文件"}), 400

        # 3. AI 深度分析 (包含评分和关键词)
        raw_result = call_qwen_analysis(resume_text, jd)

        # 4. JSON 清洗
        json_str = raw_result.strip()
        if "```json" in json_str:
            json_str = re.search(r"```json(.*?)```", json_str, re.DOTALL).group(1)

        final_data = json.loads(json_str)

        # 5. 写入缓存 (Cache Write)
        cache_manager.set(cache_key, final_data)

        return jsonify(final_data)

    except Exception as e:
        logger.exception("Error")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)