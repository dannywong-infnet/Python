import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                torch_dtype=torch.bfloat16,
                device_map="auto")
messages = [{"role":"system", "content":"você é um assistente útil."}]

def conversa(pergunta):
  messages.append({"role":"user", "content":pergunta})
  prompt = pipe.tokenizer.apply_chat_template(messages,
                                              tokenize=False,
                                              add_generation_prompt=True)
  output = pipe(
      prompt,
      max_new_tokens=500, #Quantidade maxima de palavras na resposta
      do_sample=True, #Para obter respostas variadas
      temperature=0.7, #"criatividade" do modelo (0-1)
      top_p=0.9, #limite minimo de probabilidade para que a palavra seja escolhida
      top_k=50 #quantidade de palavras candidatas a serem escolhidas
      )

  resposta = output[0]["generated_text"][len(prompt):].strip()
  messages.append({"role":"assistant","content":resposta})
  return resposta

print("*** Chatbot com TinyLlama 1.1B-Chat ***")
print("Digite 'sair' para encerrar.")

while True:
  pergunta=input("Você: ")
  if pergunta.lower() in ["sair","exit","quit"]:
    print("Encerrando o chat")
    break
  resposta = conversa(pergunta)
  print(f"TinyLlama: {resposta}")