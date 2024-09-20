import streamlit as st
import transformers
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

mc='madhavpro3/ExtractTherapyDetails'
tokenizer = AutoTokenizer.from_pretrained(mc)
mc_model = AutoModelForQuestionAnswering.from_pretrained(mc)

context = st.text_area(
	"Write input passage here",
	"The U.S. Food and Drug Administration (FDA) granted approval to atezolizumab and durvalumab in March of 2019 and 2020, respectively, for use in combination with chemotherapy for first-line treatment of patients with extensive stage small cell lung cancer. These approvals were based on data from two randomized controlled trials, IMpower133 (atezolizumab) and CASPIAN (durvalumab). Both trials demonstrated an improvement in overall survival (OS) with anti-programmed death ligand 1 antibodies when added to platinum-based chemotherapy as compared with chemotherapy alone. In IMpower133, patients receiving atezolizumab with etoposide and carboplatin demonstrated improved OS (hazard ratio [HR], 0.70; 95% confidence interval [CI], 0.54-0.91; p = .0069), with median OS of 12.3 months",
	height=300,
)

question=st.text_input(
	"Question",
	"what is the drug?",
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
  if sum(isdrugtopic)>0:
    processed_questions.append('what is the drug?')
  if sum(ispatienttopic)>0:
    processed_questions.append('who are the patients?')
  # if sum(isorrtopic)>0:
  #   processed_questions.append('what is the orr?')
  # if sum(isdortopic)>0:
  #   processed_questions.append('what is the dor?')

  return processed_questions

def generateresponse(input_question,context,model,tokenizer):
  questions_brokendown=identifytopics(input_question)
  print("questions",questions_brokendown)
  answers=[]
  if len(questions_brokendown)==0:
    answers.append("Can't find the answers you are looking for")
    return answers
  
  answers=[get_answer(question, context, model,tokenizer) for question in questions_brokendown]
  return answers

if st.button("Extract"):
  a=generateresponse(question, context, mc_model, tokenizer)
  for ans in a:
    st.write(ans)