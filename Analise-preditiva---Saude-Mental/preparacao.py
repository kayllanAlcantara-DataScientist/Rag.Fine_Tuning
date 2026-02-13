# data_prep.py
import os
import json
from transformers import AutoTokenizer

os.makedirs("knowledge_base", exist_ok=True)

text_depressao = """
Critérios Diagnósticos para Depressão Maior (resumido do DSM-5):
A. Cinco (ou mais) dos seguintes sintomas estiveram presentes durante o mesmo período de duas semanas e representam uma mudança em relação ao funcionamento anterior; pelo menos um dos sintomas é (1) humor deprimido ou (2) perda de interesse ou prazer (anedonia).
1. Humor deprimido na maior parte do dia, quase todos os dias.
2. Acentuada diminuição do interesse ou prazer em todas ou quase todas as atividades.
3. Perda ou ganho significativo de peso sem estar fazendo dieta.
4. Insônia ou hipersonia quase todos os dias.
5. Agitação ou retardo psicomotor.
6. Fadiga ou perda de energia.
7. Sentimentos de inutilidade ou culpa excessiva.
8. Capacidade diminuída de pensar ou se concentrar.
9. Pensamentos recorrentes de morte.
B. Os sintomas causam sofrimento clinicamente significativo ou prejuízo no funcionamento social, profissional ou em outras áreas importantes da vida do indivíduo.
"""
with open("knowledge_base/depressao.txt", "w", encoding="utf-8") as f:
    f.write(text_depressao)

text_ansiedade = """
Critérios Diagnósticos para Transtorno de Ansiedade Generalizada (TAG) (resumido do DSM-5):
A. Ansiedade e preocupação excessivas, ocorrendo na maioria dos dias por pelo menos seis meses, sobre diversos eventos ou atividades.
B. O indivíduo considera difícil controlar a preocupação.
C. A ansiedade e a preocupação estão associadas com três (ou mais) dos seguintes seis sintomas:
1. Inquietação ou sensação de estar com os nervos à flor da pele.
2. Fatigabilidade (cansaço fácil).
3. Dificuldade em se concentrar ou sensações de 'branco' na mente.
4. Irritabilidade.
5. Tensão muscular.
6. Perturbação do sono.
D. A ansiedade causa sofrimento clinicamente significativo.
"""
with open("knowledge_base/ansiedade.txt", "w", encoding="utf-8") as f:
    f.write(text_ansiedade)

print("Diretório 'knowledge_base' criado com sucesso.")

# --- Dados para fine-tuning (JSONL) ---
ft_template = """[CONTEXTO]
{context}
---
[PERGUNTA]
{question}
---
[INSTRUÇÃO]
Com base no contexto de critérios diagnósticos acima, formule uma resposta empática e uma pergunta de aprofundamento para o usuário. Não dê um diagnóstico, apenas explore os sintomas.
[RESPOSTA]
{answer}
"""

ctx_depressao = "Critérios Diagnósticos para Depressão Maior: ...pelo menos um dos sintomas é (1) humor deprimido ou (2) perda de interesse ou prazer (anedonia). ...6. Fadiga ou perda de energia."
ctx_ansiedade = "Critérios Diagnósticos para Transtorno de Ansiedade Generalizada (TAG): ...Ansiedade e preocupação excessivas... 1. Inquietação... 5. Tensão muscular."
ctx_depressao_completo = text_depressao

fine_tuning_data = [
    {
        "text": ft_template.format(
            context=ctx_depressao,
            question="Estou me sentindo para baixo e sem energia há semanas. Perdi o interesse em coisas que eu gostava.",
            answer="Entendo. Além do desânimo e da perda de energia, você notou alguma mudança no seu sono ou apetite recentemente?"
        )
    },
    {
        "text": ft_template.format(
            context=ctx_ansiedade,
            question="Eu me preocupo com tudo, o tempo todo. Meu coração dispara e sinto que não consigo relaxar.",
            answer="Lamento que esteja se sentindo assim. Essa preocupação constante afeta sua capacidade de concentração ou causa tensão muscular?"
        )
    },
    {
        "text": ft_template.format(
            context=ctx_depressao_completo,
            question="Não consigo dormir direito e perdi peso sem querer. Sinto-me inútil.",
            answer="Agradeço por compartilhar isso. A sensação de inutilidade é acompanhada por dificuldade de se concentrar nas tarefas do dia a dia?"
        )
    },
    {
        "text": ft_template.format(
            context="Nenhum contexto relevante encontrado.",
            question="bom dia",
            answer="Olá! Sou um assistente focado em saúde mental. Como você está se sentindo hoje?"
        )
    }
]

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
outfile = "fine_tuning_dataset.jsonl"
with open(outfile, "w", encoding="utf-8") as f:
    for entry in fine_tuning_data:
        full_text = entry["text"] + tokenizer.eos_token
        f.write(json.dumps({"text": full_text}, ensure_ascii=False) + "\n")

print(f"Arquivo '{outfile}' criado com sucesso.")
