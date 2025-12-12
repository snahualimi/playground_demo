import gradio as gr
import shutil
import requests
import json
import atexit, shutil, signal
import os
from datetime import datetime
from functions.ocr import main_ocr, DotsOCRParser
from functions.embedding import main_embed, VectorStore, EmbeddModel
from functions.guard import main_guard, GuardModel
from functions.translation import main_translate, TranslateModel
from functions.set_log import setup_logger
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

logger,_ = setup_logger()

# ===================== 模型调用函数 ===================== #

_models_loaded = False

def ocr_call(pdf):
    if not ocr_model:
        return "OCR model not initialized"
    if pdf is None:
        return "未上传 PDF"
    max_size = 10 * 1024 * 1024  # 10MB
    pdf_size = os.path.getsize(pdf.name)
    if pdf_size > max_size:
        return f"文件过大 ({(pdf_size/1024/1024):.4f}MB)，请上传 ≤10MB 的 PDF"
    os.makedirs("/home/ubuntu/playground/temp/ocr_input", exist_ok=True)
    temp_file = f"/home/ubuntu/playground/temp/ocr_input/{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    shutil.copy(pdf.name, temp_file)
    print(f'save to {os.path.abspath(temp_file)}')
    markdown_result = main_ocr(temp_file, ocr_model)

    global last_ocr_result
    last_ocr_result = markdown_result

    return  markdown_result


def embedding_call(query):
    if embed_model is None or vector_store is None:
        return 'Embedding model or vector store not initialized'
    if last_ocr_result is None:
        return "请先进行 OCR 识别，获取文本内容"
    vector_store.build_index(last_ocr_result)
    result = main_embed(query, vector_store)
    return result


def guard_call(question, answer):
    if not guard_model:
        return 'Guard model not initialized'
    result = main_guard(question, answer, guard_model)
    return result   

def translation_call(text, lang):
    if not trans_model:
        return 'Translate model not initialized'
    if lang == 'en':
        lang = 'English'
    if lang == 'zh':
        lang = 'Chinese'
    if lang == 'ar':
        lang = 'Arabic'
    result = main_translate(text, lang, trans_model)
    return result


def reset_vector_db():
    if vector_store.vectorstore is None:
        return "VectorStore 未初始化"

    vector_store.reset_db()
    logger.info("reset the vector store")
    return "🗑 VectorStore已重置"

def init_models():
    global _models_loaded
    if _models_loaded:
        logger.info("Models already loaded, skip.")
        return None
    global ocr_model, embed_model, vector_store, guard_model, trans_model,last_ocr_result
    logger.info("🚀 Loading models...")
    
    ocr_model = DotsOCRParser()
    if ocr_model is not None:
        logger.info("✅ OCR model loaded.")
    embed_model = EmbeddModel()
    vector_store = VectorStore(embed_model)    
    if vector_store is not None:
        logger.info("✅ embedding model and vector store loaded.")
    guard_model = GuardModel()     
    if guard_model is not None:
        logger.info("✅ guard model loaded.")
    trans_model = TranslateModel() 
    if trans_model is not None:
        logger.info("✅ translate model loaded.")
    _models_loaded=True
# ======================================================== #
# ====================== Gradio UI ======================== #
# ======================================================== #

with gr.Blocks(title="Model Playground") as demo:

    demo.load(init_models)

    gr.Markdown("#  Arabic Playground")

    with gr.Tabs():

        # ------------- OCR 页 ------------- #
        with gr.Tab("📄 OCR 识别"):
            gr.Markdown("**上传 PDF → 返回识别 Markdown**")
            pdf = gr.File(label="上传 PDF 文件", file_types=[".pdf"])
            ocr_btn = gr.Button("开始识别")
            ocr_output = gr.Markdown(label="OCR 输出")

            ocr_btn.click(fn=ocr_call, inputs=pdf, outputs=ocr_output)

        # ------------- Embedding + Rerank 页 ------------- #
        with gr.Tab("🔍 文档检索 Embedding") as embed_tab:
            gr.Markdown("**输入 Query  → 返回OCR识别文档检索结果**")
            query = gr.Textbox(label="Query", placeholder="输入查询语句")
            embed_btn = gr.Button("检索召回")
            embed_output = gr.JSON(label="检索结果")

            embed_btn.click(fn=embedding_call, inputs=query, outputs=embed_output)

            reset_btn = gr.Button("🗑 重置向量数据库", variant="stop")
            reset_output = gr.Markdown(label="状态")

            reset_btn.click(fn=reset_vector_db, inputs=None, outputs=reset_output)

        # ------------- Guard 页 ------------- #
        with gr.Tab("🛡 合规检测 Guard"):
            gr.Markdown("**输入问句，回复 → 判断是否合规**")
            q_in = gr.Textbox(label="提问内容")
            a_in = gr.Textbox(label="回答内容", lines=4)
            guard_btn = gr.Button("检测")
            guard_output = gr.JSON(label="合规检测结果")

            guard_btn.click(fn=guard_call, inputs=[q_in, a_in], outputs=guard_output)

        # ------------- 翻译页 ------------- #
        with gr.Tab("🌍 翻译 Translation"):
            gr.Markdown("**输入文本 → 翻译内容**")
            text = gr.Textbox(label="输入文本", lines=10)
            lang = gr.Dropdown(["en", "zh", "ar"], value="en", label="目标语言")
            trans_btn = gr.Button("翻译")
            trans_output = gr.Markdown(label="翻译结果")

            trans_btn.click(fn=translation_call, inputs=[text, lang], outputs=trans_output)

    
TEMP_DIRS = [
    "/home/ubuntu/playground/temp/vectordb",
    "/home/ubuntu/playground/temp/ocr_input",
    "/home/ubuntu/playground/temp/ocr_output"
]

def cleanup_temp():
    logger.info("\n🧹 清理临时文件中...")
    for d in TEMP_DIRS:
        try:
            if os.path.exists(d):
                shutil.rmtree(d)
                os.makedirs(d, exist_ok=True)
                logger.info(f"✔ 已清空 {d}")
        except Exception as e:
            logger.info(f"⚠ 目录清理失败 {d} → {e}")

atexit.register(cleanup_temp)
signal.signal(signal.SIGINT, lambda s,f: cleanup_temp() or exit(0))
# signal.signal(signal.SIGTERM, lambda s,f: cleanup_temp() or exit(0))

demo.launch(server_name="0.0.0.0", server_port=7861)