from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import os
import torch
from typing import List, Dict
import ollama
import base64
from io import BytesIO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FOLDERS = {
    "Quy trình quản lý VHKT": {
        "path": "data/Quy_trinh_VHKT",
        "file_type": "pdf",
        "model": "gemma3:4b",
        "faiss_index": "faiss_index_vhkt"
    },
    "Quy định An toàn thông tin": {
        "path": "data/Quyet_dinh_ATTT",
        "file_type": "pdf",
        "model": "gemma3:4b",
        "faiss_index": "faiss_index_attt"
    },
    "Quy trình/Quy định khác": {
        "path": "data/Cloud",
        "file_type": "csv",
        "model": "gemma3:4b",
        "faiss_index": "faiss_index_cloud"
    },
}

vectorstores = {}
retrievers = {}
message_stores = {}

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": "mps" if torch.backends.mps.is_available() else "cpu"}
)

system_prompt = (
    "Bạn là AINoiBo, một mô hình ngôn ngữ lớn, được tạo ra bởi Chuyên viên Vũ Phát Đạt - Phòng Kỹ thuật khai thác - Trung tâm MDS. "
    "Bạn là một trợ lý ảo, giúp trả lời về những câu hỏi nội bộ trong tài liệu, vì thế chỉ cung cấp thông tin có trong tài liệu, không được cung cấp gì khác ngoài tài liệu. "
    "Chỉ trả lời dựa trên tài liệu được cung cấp hoặc nội dung hình ảnh được gửi, không sử dụng kiến thức bên ngoài. "
    "TUYỆT ĐỐI KHÔNG ĐƯỢC SỬ DỤNG KIẾN THỨC NÀO KHÁC NGOÀI TÀI LIỆU HOẶC HÌNH ẢNH, NẾU KHÔNG CÓ THÔNG TIN HÃY NÓI KHÔNG BIẾT. "
    "Hãy trả lời ngắn gọn, súc tích, không dài dòng. "
    "Không giải thích gì cả, hãy đưa ra câu trả lời luôn. "
    "TUYỆT ĐỐI CHỈ ĐƯỢC TRẢ LỜI BẰNG TIẾNG VIỆT, NGHIÊM CẤM KHÔNG ĐƯỢC SỬ DỤNG NGÔN NGỮ KHÁC. "
    "TẤT CẢ CÁC PHẢN HỒI PHẢI CHỈ SỬ DỤNG TIẾNG VIỆT, KHÔNG ĐƯỢC TRẢ LỜI BẰNG NGÔN NGỮ KHÁC. "
    "Ghi nguồn tài liệu sau mỗi câu trả lời nếu câu trả lời dựa trên tài liệu.\n\n{context}"
)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

suggestion_prompt = ChatPromptTemplate.from_messages([
    ("system", "Dựa trên lịch sử trò chuyện và nội dung tài liệu, tạo câu hỏi gợi ý ngắn gọn, cụ thể, không lặp lại. "
               "TẤT CẢ CÁC PHẢN HỒI PHẢI CHỈ SỬ DỤNG TIẾNG VIỆT, KHÔNG ĐƯỢC TRẢ LỜI BẰNG NGÔN NGỮ KHÁC. "
               "Trả lời dưới dạng danh sách, mỗi câu một dòng.\n\nLịch sử: {history}\nTài liệu: {context}"),
    ("human", "Tạo gợi ý câu hỏi.")
])

def load_documents(folder_path: str, file_type: str) -> List:
    documents = []
    if not os.path.exists(folder_path):
        raise ValueError(f"Thư mục {folder_path} không tồn tại")
    if file_type == "csv":
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".csv"):
                loader = CSVLoader(file_path=os.path.join(folder_path, file_name), encoding='utf-8')
                documents.extend(loader.load())
    elif file_type == "pdf":
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".pdf"):
                loader = PyPDFLoader(file_path=os.path.join(folder_path, file_name))
                documents.extend(loader.load())
    return documents

def initialize_vectorstore(category: str):
    folder_info = FOLDERS[category]
    faiss_index = folder_info["faiss_index"]
    
    if os.path.exists(faiss_index):
        vectorstore = FAISS.load_local(faiss_index, embeddings, allow_dangerous_deserialization=True)
    else:
        documents = load_documents(folder_info["path"], folder_info["file_type"])
        if not documents:
            raise ValueError(f"Không tìm thấy tài liệu trong {category}")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        all_splits = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(all_splits, embeddings)
        vectorstore.save_local(faiss_index)
    
    vectorstores[category] = vectorstore
    retrievers[category] = vectorstore.as_retriever(search_type='similarity', search_kwargs={"k": 10})

for category in FOLDERS.keys():
    try:
        initialize_vectorstore(category)
    except Exception as e:
        print(f"Lỗi khi khởi tạo vectorstore cho {category}: {str(e)}")

def get_session_history(session_id: str, category: str) -> ChatMessageHistory:
    key = f"{category}_{session_id}"
    if key not in message_stores:
        message_stores[key] = ChatMessageHistory()
    return message_stores[key]

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    suggestions: List[str]

# Endpoint /chat (sửa để nhận FormData thay vì JSON)
@app.post("/chat", response_model=ChatResponse)
async def chat(
    session_id: str = Form(...),
    category: str = Form(...),
    question: str = Form(default="")
):
    if category not in FOLDERS:
        raise HTTPException(status_code=400, detail="Danh mục không hợp lệ")

    model_name = FOLDERS[category]["model"]
    llm = ChatOllama(model=model_name, base_url='http://localhost:11434')

    retriever = retrievers.get(category)
    if not retriever:
        raise HTTPException(status_code=500, detail="Retriever chưa được khởi tạo")

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    conversational_rag_chain = RunnableWithMessageHistory(
        runnable=rag_chain,
        get_session_history=lambda: get_session_history(session_id, category),
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    identity_questions = ["bạn là ai", "bạn là gì"]
    is_identity_question = any(q in question.lower() for q in identity_questions)
    
    if is_identity_question:
        answer = (
            "Tôi là AINoiBo, một mô hình ngôn ngữ lớn được tạo ra bởi Chuyên viên Vũ Phát Đạt - "
            "Phòng Kỹ thuật khai thác - Trung tâm MDS."
        )
        sources = []
        retrieved_docs = []
    else:
        retrieved_docs = retriever.invoke(question)
        if not retrieved_docs:
            answer = "Không tìm thấy thông tin trong tài liệu."
            sources = []
        else:
            response = conversational_rag_chain.invoke(
                {"input": question, "context": retrieved_docs},
                {"configurable": {"session_id": session_id}}
            )
            answer = response["answer"]
            answer = answer.strip().replace("\n", " ").replace("[^\\p{L}]+", " ").strip()
            sources = list(set(os.path.basename(doc.metadata.get('source', 'Nguồn không xác định')) for doc in retrieved_docs))

    history = get_session_history(session_id, category).messages
    context = "\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else ""
    suggestion_chain = suggestion_prompt | llm
    try:
        suggestion_response = suggestion_chain.invoke({
            "history": str(history),
            "context": context
        })
        suggestions = suggestion_response.content.strip().split('\n')[:3]
        suggestions = [s.strip() for s in suggestions if s.strip()]
    except:
        suggestions = [
            "Bạn có thể hỏi thêm về chủ đề này không?",
            "Có thông tin nào liên quan khác không?",
            "Bạn có muốn giải thích chi tiết hơn không?"
        ]

    return ChatResponse(answer=answer, sources=sources, suggestions=suggestions)

# Endpoint /chat_with_image (giữ nguyên vì đã đúng)
@app.post("/chat_with_image", response_model=ChatResponse)
async def chat_with_image(
    session_id: str = Form(...),
    category: str = Form(...),
    question: str = Form(default=""),
    image: UploadFile = File(None)
):
    if category not in FOLDERS:
        raise HTTPException(status_code=400, detail="Danh mục không hợp lệ")

    model_name = FOLDERS[category]["model"]
    retriever = retrievers.get(category)
    if not retriever:
        raise HTTPException(status_code=500, detail="Retriever chưa được khởi tạo")

    # Khởi tạo lịch sử trò chuyện
    history = get_session_history(session_id, category).messages

    # Nếu có hình ảnh
    image_base64 = None
    if image:
        image_data = await image.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

    # Nếu có câu hỏi, tìm kiếm tài liệu liên quan
    retrieved_docs = []
    if question:
        retrieved_docs = retriever.invoke(question)

    # Chuẩn bị context từ tài liệu
    context = "\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else ""
    sources = list(set(os.path.basename(doc.metadata.get('source', 'Nguồn không xác định')) for doc in retrieved_docs)) if retrieved_docs else []

    # Chuẩn bị tin nhắn gửi đến Ollama
    messages = [{"role": "system", "content": system_prompt.format(context=context)}]
    for msg in history:
        messages.append({"role": "user" if msg.type == "human" else "assistant", "content": msg.content})
    
    # Thêm câu hỏi và hình ảnh (nếu có)
    if image_base64:
        messages.append({
            "role": "user",
            "content": question or "Hình ảnh này có gì?",
            "images": [image_base64]
        })
    else:
        messages.append({"role": "user", "content": question})

    # Gọi Ollama API
    try:
        response = ollama.chat(model=model_name, messages=messages)
        answer = response['message']['content']
        answer = answer.strip().replace("\n", " ").replace("[^\\p{L}]+", " ").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi gọi Ollama: {str(e)}")

    # Thêm câu hỏi và câu trả lời vào lịch sử
    get_session_history(session_id, category).add_user_message(question or "Hình ảnh này có gì?")
    get_session_history(session_id, category).add_ai_message(answer)

    # Tạo gợi ý câu hỏi
    llm = ChatOllama(model=model_name, base_url='http://localhost:11434')
    suggestion_chain = suggestion_prompt | llm
    try:
        suggestion_response = suggestion_chain.invoke({
            "history": str(history),
            "context": context
        })
        suggestions = suggestion_response.content.strip().split('\n')[:3]
        suggestions = [s.strip() for s in suggestions if s.strip()]
    except:
        suggestions = [
            "Bạn có thể hỏi thêm về chủ đề này không?",
            "Có thông tin nào liên quan khác không?",
            "Bạn có muốn giải thích chi tiết hơn không?"
        ]

    return ChatResponse(answer=answer, sources=sources, suggestions=suggestions)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)