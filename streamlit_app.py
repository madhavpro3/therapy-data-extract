import streamlit as st
import transformers
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
import pandas as pd

mc='madhavpro3/ExtractTherapyDetails'
tokenizer = AutoTokenizer.from_pretrained(mc)
mc_model = AutoModelForQuestionAnswering.from_pretrained(mc)

st.header("Input FDA approval passages to extract drug information")

passages = st.text_area(
	"Input passage(s)",
	placeholder="""  The U.S. Food and Drug Administration (FDA) granted approval to atezolizumab and durvalumab...
   
  In April 2022, the FDA approved axicabtagene ciloleucel (axi-cel) for adults with large B-cell lymphoma (LBCL) ....""",

	height=250,
)

question=st.text_input(
	"Question",
   placeholder="Try: what is the drug? who are the patients? what is the ORR?",
   key="ques"
)

def get_answer(question, context, model,tokenizer):
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt",max_length=512,truncation=True)
    start_logits, end_logits = model(**inputs).values()


    start_index_and_logits = torch.argmax(start_logits, dim=1).item(), start_logits[0].max().item()
    end_index_and_logits = torch.argmax(end_logits, dim=1).item(), end_logits[0].max().item()

    if end_index_and_logits[0] >= start_index_and_logits[0]:
        start_index, end_index = start_index_and_logits[0], end_index_and_logits[0]
    else:
        if start_index_and_logits[1] > end_index_and_logits[1]:
            start_index, end_index = start_index_and_logits[0], start_index_and_logits[0]
        else:
            start_index, end_index = end_index_and_logits[0], end_index_and_logits[0]


    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index+1]))
    return answer


def identifytopics(input_question):
  drug=['drug','therapy']
  patients=['patients']
  # orr=['objective response', 'overall response', 'orr']
  # dor=['duration','response duration','dor']
  input_question=input_question.lower()

  isdrugtopic=[input_question.count(drugtopic)>0 for drugtopic in drug]
  ispatienttopic=[input_question.count(patienttopic)>0 for patienttopic in patients]
  # isorrtopic=[input_question.count(orrtopic)>0 for orrtopic in orr]
  # isdortopic=[input_question.count(dortopic)>0 for dortopic in dor]

  processed_questions=[]
  topics=[]
  if sum(isdrugtopic)>0:
    processed_questions.append('what is the drug?')
    topics.append("Drug")
  if sum(ispatienttopic)>0:
    processed_questions.append('who are the patients?')
    topics.append("Patients")
  # if sum(isorrtopic)>0:
  #   processed_questions.append('what is the orr?')
  # if sum(isdortopic)>0:
  #   processed_questions.append('what is the dor?')

  return processed_questions,topics

def generateresponse(input_question,passages,model,tokenizer):
  passages_lst=passages.split('\n\n')
  questions_lst,topics=identifytopics(input_question)
  # print("questions",questions_brokendown)
  answers=[]
  if len(questions_lst)==0:
    answers.append("Can't find the answers you are looking for")
    return answers,topics
  
  answers=[[get_answer(question, passage, model,tokenizer) 
            for question in questions_lst] for passage in passages_lst]
  return answers,topics

if st.button("Answer this"):
  question=st.session_state.ques
  a,topics=generateresponse(question, passages, mc_model, tokenizer)
  ans_df=pd.DataFrame(a,columns=topics)   
  st.write(ans_df)
if st.button("Extract all"):
  question="drug patients orr dor"
  a,topics=generateresponse(question, passages, mc_model, tokenizer)
  ans_df=pd.DataFrame(a,columns=topics)   
  st.write(ans_df)
     
# col1,col2=st.columns(2)

# with col1:
#    st.button("Answer this",on_click=viewresults)
# with col2:
#    question="drug patients orr dor"
#    st.button("Extract all",on_click=viewresults)


# def viewresults():
#   a,topics=generateresponse(question, passages, mc_model, tokenizer)
#   ans_df=pd.DataFrame(a,columns=topics)   
#   st.write(ans_df)

# with col1:
#   st.button("Answer this",on_click=viewresults):
#     a,topics=generateresponse(question, passages, mc_model, tokenizer)
#     ans_df=pd.DataFrame(a,columns=topics)
#     viewresults(ans_df)
#     # st.write(ans_df)

# with col2:
#   if st.button("Extract all"):
#     question="drug patients ORR DoR"
#     # ans_df=pd.DataFrame({"Drug":a[0],"Patients":a[1]})
#     ans_df=pd.DataFrame(a,columns=topics)
#     viewresults(ans_df)
#     # st.write(ans_df)

