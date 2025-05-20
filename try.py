#%%
import streamlit as st
from dotenv import load_dotenv
# from htmlTemplates import css
from langchain.vectorstores import Chroma
from model import AzureModel
import glob
import os
from langchain.text_splitter import CharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PyMuPDFLoader # 解決 pdf 亂碼問題
from langchain.document_loaders.csv_loader import CSVLoader
import subprocess
#%%

### find pdf ###
def find_pdf(path):
    pdf_docs=[]
    for file in glob.glob(os.path.join(path, "*.pdf")):
        pdf_docs.append(file)
    return(pdf_docs)

### find csv ###
def find_csv(path):
    csv_docs=[]
    for file in glob.glob(os.path.join(path, "*.csv")):
        csv_docs.append(file)
    return(csv_docs)

### split chunk(pdf) ###
def get_texts_pdf(pdf_docs):
    docs=[]
    # loading pdf 
    for path in pdf_docs:
        # docs.extend(PyPDFLoader(path).load())
        docs.extend(PyMuPDFLoader(path).load())

    # split chunk
    text_splitter = CharacterTextSplitter(
        ##  separator=",",
        chunk_size=1000,
        chunk_overlap=300,
        length_function=len
    )
    texts = text_splitter.split_documents(docs)
    return texts

### split chunk(csv) ###
def get_texts_csv(csv_docs):
    docs=[]
    # loading csv 
    for path in csv_docs:
        docs.extend(CSVLoader(path).load())
    # split chunk
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=300,
        length_function=len
    )
    texts = text_splitter.split_documents(docs)
    return texts

@st.cache_resource
def savings_vectordb(_texts, _adda):
    # create vectordb
    db = Chroma.from_documents(_texts , _adda)
    return db

### query similarity text ###
def query_vectordb(_question, _db):
    msg_text= _db.similarity_search(_question, k=10)
    return(msg_text)

def get_source(texts):
    s_list=[]
    for i in texts:
        s_list.append(i.metadata['source'])
    return set(s_list)

# def deal_filename(path_doc):
#     for i in path_doc:

#         # 以 '/' 分割路徑得到各個部分
#         pathchuck = i.split("/")
#         # 獲取沒有附檔名的文件名稱
#         pathname = pathchuck[-1].replace(".pdf", "")
#         # 有反斜線的部分做特殊處理ㄠ
#         if "\\" in i:
#             name = i.split("\\")[-1].split("/")[-1].replace(".pdf", "")
#         else:
#             # 如果沒有反斜線，直接使用 pathname`
#             name = pathname

#         if "2024永續報告書" in i:
#             st.markdown(f'* ### [{name}](https://www.transglobe.com.tw/campaign/esg/campaign/2024CSreport/2024ESGreportCH.pdf)')


# def open_file(path):
#     pdf_file_path = path
#     #打開 PDF 檔案
#     subprocess.run(['open', pdf_file_path])

def main():

    # 模型建構
    st.session_state.Az = AzureModel()

    # 載入系統變數
    load_dotenv()

    st.set_page_config(page_title="全球人壽智能助理",
                       page_icon=":rocket:")

    ## st.write(css, unsafe_allow_html=True) # 有寫沒寫好像沒差？

    st.image(os.getenv("IMAGE_PATH"), use_column_width=True)

    # 中間開頭畫面
    st.header("全球人壽智能助理")

    # 中間機器人歡迎對話
    with st.chat_message("assistant"):
        st.markdown("##### 我是全球人壽智能助理，請問有什麼能為您協助的呢？")

    # 下方輸入格
    user_question = st.chat_input("輸入您的問題...")

    # 按鈕判斷初始值
    st.session_state.chat=0

    # 常見問題按鈕畫面
    if st.button("我提出理賠申請後，保險金多久會下來？"):
        user_question = "我提出理賠申請後，保險金多久會下來？"
        st.session_state.chat=1
    if st.button("健康告知書相關表單下載？"):
        user_question = "健康告知書相關表單下載？"
    if st.button("2023年獲得哪些永續獎項？"):
        user_question = "2023年獲得哪些永續獎項？"
        st.session_state.chat=1
    if st.button("你支持哪一個政黨？"):
        user_question = "你支持哪一個政黨？"
        st.session_state.chat=1
    # if st.button("品牌價值與永續發展策略三大主軸？"):
    #     user_question = "品牌價值與永續發展策略三大主軸？"
    #     st.session_state.chat=1
    # if st.button("保險事故發生後多久要提出理賠申請？"):
    #     user_question = "保險事故發生後多久要提出理賠申請？"
    #     st.session_state.chat=1

    # 只是個初始化
    if "messages" not in st.session_state.keys():
        st.session_state.messages = []
    else:
        # 重新畫所有訊息
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # 中間畫面-機器人與使用者對話
    if user_question or st.session_state.chat==1: # 只要寫一個就好吧？
        # 按鈕判斷初始化
        st.session_state.chat=0

        # 使用者問題訊息
        with st.chat_message("user"):
            st.markdown("##### "+user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        # vectoedb 前十相似文本
        st.session_state.texts = query_vectordb(user_question, st.session_state.db)

        # vectoedb 前三相似文本來源
        # st.session_state.source = get_source(st.session_state.texts)

        # 機器人回覆訊息
        with st.spinner("##### 正在為您尋找中，請稍等～"):
            type = AzureModel().classification(user_question, st.session_state.texts)
            if type == "保險相關":
                ans = st.session_state.Az.insurance_gpt(user_question, st.session_state.texts)
            elif type == "永續發展相關":
                ans = st.session_state.Az.sustainable_gpt(user_question, st.session_state.texts)
            else:
                ans = "抱歉，我沒辦法回答您～"


            # gpt回傳
            ## st.session_state.gpt = st.session_state.Az.using_gpt(user_question, st.session_state.texts)

            # gpt回傳(生成內容)
            ## st.session_state.msg = st.session_state.gpt[0]
            ## st.session_state.msg = st.session_state.gpt
            ## st.session_state.msg = ans

            # gpt回傳(生成花費)(小數點第三位)
            ## st.session_state.cost =f"{st.session_state.gpt[1]:.3f}"

            with st.chat_message("assistant"):
                # 生成內容
                st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})

                # 花費(小數點第三位)
                ## st.markdown("本次生成花費："+st.session_state.cost+" TWD")

    # print("IMAGE_PATH:",os.getenv("IMAGE_PATH"))
    # print("PDF_PATH：",os.getenv("PDF_PATH"))

    # 畫面最左邊          
    with st.sidebar:
        st.image(os.getenv("IMAGE_PATH"))
        with st.spinner("載入中"):
            #找 pdf 檔案
            path_doc_pdf = find_pdf(os.getenv("PDF_PATH"))
            path_doc_csv = find_csv(os.getenv("CSV_PATH"))

            #Linux
            # path_doc1 = ['2024永續報告書.pdf']
            # print("path_doc:",path_doc)
            st.markdown("## 以下是可供查詢的內容：")
            st.markdown(f'* ### [{"2024 永續報告書.pdf"}](https://www.transglobe.com.tw/campaign/esg/campaign/2024CSreport/2024ESGreportCH.pdf)')
            st.markdown(f'* ### [{"2023 永續報告書.pdf"}](https://www.transglobe.com.tw/campaign/2023CSreport/2023ESGreportCH.pdf)')
            # print("path_doc:", path_doc1)
            # deal_filename(path_doc1)

            # 切檔案
            st.session_state.docs_pdf = get_texts_pdf(path_doc_pdf)
            st.session_state.docs_csv = get_texts_csv(path_doc_csv)

            # 建立db庫
            st.session_state.db = savings_vectordb(st.session_state.docs_pdf+st.session_state.docs_csv, st.session_state.Az.using_embedding())
    
        # st.success("")

    # 只是個初始化
    # if "messages" not in st.session_state.keys():
    #     st.session_state.messages = []
    # else:
    #     # 存訊息（給下次重畫用）
    #     st.session_state.messages.append({"role": "user", "content": user_question})
    #     st.session_state.messages.append({"role": "assistant", "content": ans})

    #     # 重新畫所有訊息
    #     for message in st.session_state.messages:
    #         with st.chat_message(message["role"]):
    #             st.markdown(message["content"])
    
    


if __name__ == "__main__":
    main()
