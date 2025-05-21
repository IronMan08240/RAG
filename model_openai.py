#%%
import os 
import openai
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
#%%

# 系統變數
load_dotenv()

class AzureModel:

    def __init__(self):
        self.base = os.getenv("OPENAI_API_BASE")
        self.key = os.getenv("OPENAI_API_KEY")
        self.version = os.getenv("OPENAI_API_VERSION")
        self.name = os.getenv("GPT3_ENGINE")

    # 查看模型名稱
    def get_name(self):
        return(self.name)
        
    # 設定 Azure OpenAI 參數
    def setmodel(self):
        openai.api_base = self.base
        openai.api_key = self.key
        openai.api_version = self.version
        openai.api_type = "azure"

    ### Embedding Model ###
    def using_embedding(self):
        self.name = os.getenv("Embedding_DEPLOYMENT")
        self.setmodel()
        embeddings = OpenAIEmbeddings(deployment=self.name)
        return(embeddings)

    ### 保險模型 ###
    def insurance_gpt(self,question,text):

        # 載入 Azure OpenAI 參數
        self.setmodel()

        # 設定 prompt
        prompt = f"""
        0. 開頭先回答這是一個根據全球人壽官方網站的回答。
        1. 你是一個文字摘要機器人，繁體中文是你唯一的回答語言，不要進行數字計算，每段分行，條列式說明。\n
        2. 生成結束後只要是中文字都要檢查是否用繁體中文。\n
        3. 兩組三個單引用號符號中的 question 是業務的疑問，只能根據疑問去尖角括號中的 text 進行文字參考生成。\n
        4. 詢問表單下載相關的問題，除了回答問題，若資料庫有對應的表單名稱，則附上所有表單名稱所對應的相關表單名稱及連結；若資料庫沒有對應的表單名稱，則不需附上任何名稱或連結。\n
        5. 條列若少於兩條，則不需使用條列式回答問題。\n
        6. 跟政治相關的問題，請回答我只是一個智能助理機器人無法回答你政治相關的問題，但你可以詢問我跟目前可供查詢的商品有關的問題。\n
        7. 跟種族歧視、性暗示、不雅字眼，或違反聯合國人權相關的問題，請回答我只是一個智能助理機器人無法回答你這個的問題，但你可以詢問我跟目前可供查詢的商品有關的問題。\n
        8. 若問題與資料庫的內容無關，請回答我只是一個智能助理機器人無法回答你這個的問題。\n
        Question: '''{question}'''\n
        Text: <{text}> \n
        """
        
        # 啟用模型
        message = [
            {'role':'system','content':prompt},
            {'role':'user', 'content':question}
        ]

        response = openai.ChatCompletion.create(
            engine=self.name,
            messages= message,
            temperature=0
        )

        # 模型回覆訊息
        showmsg = response.choices[0].message["content"]
        
        return(showmsg)
    
    ### 永續模型 ###
    def sustainable_gpt(self,question,text):

        # 載入 Azure OpenAI 參數
        self.setmodel()

        # 設定 prompt
        prompt = f"""
        0. 開頭先回答這是一個根據全球人壽永續報告書的回答。
        1. 你是一個文字摘要機器人，繁體中文是你唯一的回答語言，不要進行數字計算，每段分行，條列式說明。\n
        2. 生成結束後只要是中文字都要檢查是否用繁體中文。\n
        3. 兩組三個單引用號符號中的 question 是業務的疑問，根據疑問去尖角括號中的 text，進行文字參考生成。\n
        4. 條列若少於兩條，則不需使用條列式回答問題。\n
        5. 跟政治相關的問題，請回答我只是一個智能助理機器人無法回答你政治相關的問題，\n
           但你可以詢問我跟目前可供查詢的商品有關的問題。\n
        6. 跟種族歧視、性暗示、不雅字眼，或違反聯合國人權相關的問題，\n
           請回答我只是一個智能助理機器人無法回答你這個的問題，但你可以詢問我跟目前可供查詢的商品有關的問題。\n
        7. 若問題與資料庫的內容無關，請回答我只是一個智能助理機器人無法回答你這個的問題。\n
        8. 在最後給出資料來源。\n
        9. 若問題沒有指定年份，但資料內具有年份資訊，則回答以條列的方式，分成各個年份回答。\n
        Question: '''{question}'''\n
        Text: <{text}> \n
        """
        
        # 啟用模型
        message = [
            {'role':'system','content':prompt},
            {'role':'user', 'content':question}
        ]

        response = openai.ChatCompletion.create(
            engine=self.name,
            messages= message,
            temperature=0
        )

        # 模型回覆訊息
        showmsg = response.choices[0].message["content"]
        return(showmsg)
    
    ### 判斷該使用何種模型 ###
    def classification(self, question, text):
        self.setmodel()
        prompt = f"""
        1. 兩組三個單引用號符號中的 question 是使用者的疑問，根據使用者疑問、去尖角括號中的text和參考資料來源，判斷這是「永續發展相關」、「保險相關」及「其它」中的哪一類問答。\n
        2. 回答只能是「永續發展相關」、「保險相關」及「其它」其中一個單詞。\n
        Question: '''{question}'''\n
        Text: <{text}> \n
        """
        message = [
            {'role':'system','content':prompt},
            {'role':'user', 'content':question}
        ]
        response = openai.ChatCompletion.create(
            engine=self.name,
            messages= message,
            temperature=0
        )
        showmsg = response.choices[0].message["content"]
        return(showmsg)
    
    
