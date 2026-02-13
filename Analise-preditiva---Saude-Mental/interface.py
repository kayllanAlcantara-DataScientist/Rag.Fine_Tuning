# app_streamlit.py
import streamlit as st
from langchain_community.llms import Ollama
import json
import os

# --- Configuração da Página ---
st.set_page_config(
    page_title="Auxiliar de Saúde Mental",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Estilo CSS (O mesmo que você tinha) ---
st.markdown("""
    <style>
    h1 { text-align: center; color: #31333F; font-weight: 600; }
    p[data-testid="stCaption"] { text-align: center; font-style: italic; }
    div[data-testid="stChatMessage"] { border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.04); background-color: #FFFFFF; border: 1px solid #E0E0E0; }
    div[data-testid="stChatMessage"][data-user-message="true"] { background-color: #F0F2F6; border: 1px solid #D1D1D1; }
    div[data-testid="stWarning"] { border-radius: 8px; border-color: #FFC107; background-color: #FFFBEA; }
    button[data-testid="baseButton-secondary"] { border-radius: 8px; }
    div[data-testid="stRadio"] label { font-size: 1.1em; font-weight: 500; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- Carregamento do Modelo ---
@st.cache_resource
def load_llm():
    try:
        # Use um modelo um pouco maior se possível para melhor análise, 
        # mas o qwen2.5:3b pode funcionar.
        return Ollama(model="qwen2.5:3b") 
    except Exception as e:
        st.error(f"Erro ao carregar modelo Ollama: {e}")
        return None

llm = load_llm()

# --- DEFINIÇÃO DO QUESTIONÁRIO ---
# Aqui definimos as 10 perguntas, opções e a qual área elas se ligam.
# Esta estrutura é a "lógica" que o LLM usará depois.
QUESTIONARIO = [
    {
        "pergunta": "Q1: Nas últimas duas semanas, com que frequência você sentiu pouco interesse ou prazer em fazer as coisas?",
        "opcoes": ["Nenhuma", "Vários dias", "Mais da metade dos dias", "Quase todos os dias"],
        "area": "Depressão (Anedonia)"
    },
    {
        "pergunta": "Q2: Nas últimas duas semanas, com que frequência você se sentiu nervoso(a), ansioso(a) ou 'com os nervos à flor da pele'?",
        "opcoes": ["Nenhuma", "Vários dias", "Mais da metade dos dias", "Quase todos os dias"],
        "area": "Ansiedade (Preocupação)"
    },
    {
        "pergunta": "Q3: Com que frequência você tem dificuldade para manter o foco em tarefas ou conversas?",
        "opcoes": ["Raramente", "Às vezes", "Frequentemente", "Quase sempre"],
        "area": "TDAH (Desatenção)"
    },
    {
        "pergunta": "Q4: Você já teve períodos distintos (dias ou semanas) em que se sentiu 'no topo do mundo', com muito mais energia ou autoconfiança do que o normal, a ponto de amigos ou familiares notarem?",
        "opcoes": ["Não, nunca", "Acho que sim, mas foi breve", "Sim, claramente"],
        "area": "Bipolaridade (Mania/Hipomania)"
    },
    {
        "pergunta": "Q5: Nas últimas duas semanas, com que frequência você se sentiu 'para baixo', deprimido(a) ou sem esperança?",
        "opcoes": ["Nenhuma", "Vários dias", "Mais da metade dos dias", "Quase todos os dias"],
        "area": "Depressão (Humor)"
    },
    {
        "pergunta": "Q6: Nas últimas duas semanas, com que frequência você esteve tão inquieto(a) ou agitado(a) que era difícil ficar parado(a)?",
        "opcoes": ["Nenhuma", "Vários dias", "Mais da metade dos dias", "Quase todos os dias"],
        "area": "Ansiedade (Física) / TDAH (Hiperatividade)"
    },
    {
        "pergunta": "Q7: Com que frequência você age por impulso (toma decisões rápidas sem pensar nas consequências)?",
        "opcoes": ["Raramente", "Às vezes", "Frequentemente", "Quase sempre"],
        "area": "TDAH (Impulsividade)"
    },
    {
        "pergunta": "Q8: Você já teve períodos em que precisava dormir muito menos do que o habitual (ex: 2-3 horas por noite) mas, mesmo assim, se sentia cheio de energia e não cansado?",
        "opcoes": ["Não, nunca", "Acho que sim, mas ainda me sentia cansado(a)", "Sim, e eu não sentia cansaço"],
        "area": "Bipolaridade (Sono/Mania)"
    },
    {
        "pergunta": "Q9: Nas últimas duas semanas, com que frequência você se sentiu mal sobre si mesmo(a) — ou que era um fracasso ou que tinha decepcionado a si mesmo(a) ou sua família?",
        "opcoes": ["Nenhuma", "Vários dias", "Mais da metade dos dias", "Quase todos os dias"],
        "area": "Depressão (Culpa/Autoestima)"
    },
    {
        "pergunta": "Q10: Quão difícil é para você organizar tarefas e atividades do dia-a-dia?",
        "opcoes": ["Nada difícil", "Um pouco difícil", "Muito difícil", "Extremamente difícil"],
        "area": "TDAH (Organização/Função Executiva)"
    }
]

# --- PROMPT DE ANÁLISE FINAL ---
# (Cole aqui o PROMPT_ANALISE_FINAL que defini na seção anterior)
PROMPT_ANALISE_FINAL = """
Você é um assistente de IA especializado em psicologia, agindo como uma ferramenta de pré-triagem para um profissional de saúde mental.

Sua tarefa é analisar as 10 respostas de um questionário de triagem e fornecer um resumo conciso (um "mini-diagnóstico" ou "resumo de triagem") para o psicólogo.

NÃO FAÇA NOVAS PERGUNTAS. Apenas forneça a análise.

### Lógica de Triagem (Regras para sua análise):

Você deve procurar por padrões que sugiram tendências para quatro áreas principais. Use esta lógica para fundamentar sua análise:

1.  **Indicadores de Depressão:**
    * **Perguntas-Chave:** Q1 (Anedonia/Perda de Interesse), Q5 (Humor Deprimido).
    * **Lógica:** Respostas como "Nenhuma" ou "Quase todos os dias" nestas perguntas são indicadores fortes. A Q9 (Sentimento de culpa) também é um indicador de suporte.

2.  **Indicadores de Ansiedade (TAG):**
    * **Perguntas-Chave:** Q2 (Preocupação Excessiva), Q6 (Sintomas Físicos / Inquietação).
    * **Lógica:** Respostas como "Vários dias" ou "Quase todos os dias" nestas perguntas sugerem ansiedade generalizada.

3.  **Indicadores de TDAH (Desatenção/Hiperatividade):**
    * **Perguntas-Chave:** Q3 (Dificuldade de Foco), Q7 (Inquietação/Impulsividade), Q10 (Dificuldade de Organização).
    * **Lógica:** Respostas como "Frequentemente" ou "Quase sempre" nestas perguntas indicam dificuldades consistentes de atenção e/ANA ou hiperatividade.

4.  **Indicadores de Bipolaridade (Episódios de Mania/Hipomania):**
    * **Perguntas-Chave:** Q4 (Picos de Energia/Grandiosidade), Q8 (Redução da Necessidade de Sono).
    * **Lógica:** Respostas "Sim" nestas perguntas são indicadores muito distintos e prioritários. Episódios de energia extrema combinados com pouca necessidade de sono (sem sentir cansaço) são sinais clássicos.

### Respostas do Paciente:

{RESPOSTAS_DO_USUARIO}

### Formato da Saída (Obrigatório):

Gere a resposta EXATAMENTE neste formato:

**Resumo de Triagem para Análise Profissional**

**Áreas de Destaque:**
* **[Área 1, ex: Depressão]:** [Explicação concisa do porquê, citando as respostas-chave. Ex: "Indicado por humor deprimido quase diário (Q5) e perda total de interesse (Q1)."]
* **[Área 2, ex: Ansiedade]:** [Explicação concisa do porquê, citando as respostas-chave.]
* **(Se nenhuma área se destacar, escreva: "Nenhuma área de destaque primário identificada.")**

**Observações Adicionais:**
[Um breve parágrafo (2-3 frases) resumindo a impressão geral. Por exemplo: "O paciente relata uma sobreposição de sintomas ansiosos e depressivos, com forte ênfase na dificuldade de organização (Q10)."]

**Recomendação de Triagem:**
[Ex: "Sugere-se priorizar a avaliação para Transtorno Depressivo Maior e Transtorno de Ansiedade Generalizada."]
"""


# --- Inicialização do Estado da Aplicação ---
if "current_question" not in st.session_state:
    st.session_state.current_question = 0  # Índice da pergunta atual
if "answers" not in st.session_state:
    st.session_state.answers = {} # Dicionário para guardar as respostas
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

# --- Cabeçalho ---
st.title("Auxiliar de Saúde Mental")
st.caption("Protótipo para Fins Acadêmicos")

if not st.session_state.analysis_complete:
    st.markdown("""
    Este é um questionário de triagem acadêmico. 
    Por favor, responda às 10 perguntas a seguir com base em como você se sentiu **nas últimas duas semanas**.
    """)
st.warning("⚠️ Atenção: Esta ferramenta NÃO substitui um diagnóstico profissional. É apenas uma demonstração tecnológica.")
st.divider()

# --- Lógica do Questionário ---
if llm is None:
    st.error("O modelo de IA não pôde ser carregado. Verifique a conexão com o Ollama.")
else:
    # Se ainda não terminamos as perguntas
    if st.session_state.current_question < len(QUESTIONARIO):
        
        # Pega a pergunta atual
        q_data = QUESTIONARIO[st.session_state.current_question]
        
        # Mostra a pergunta e as opções
        st.subheader(f"Pergunta {st.session_state.current_question + 1} de {len(QUESTIONARIO)}")
        resposta = st.radio(
            q_data["pergunta"],
            options=q_data["opcoes"],
            key=f"q_{st.session_state.current_question}"
        )
        
        # Botão para ir para a próxima pergunta
        if st.button("Próximo", type="primary"):
            # Salva a resposta
            st.session_state.answers[q_data["pergunta"]] = resposta
            # Avança para a próxima pergunta
            st.session_state.current_question += 1
            st.rerun() # Recarrega a página para mostrar a próxima pergunta

    # Se terminamos as perguntas, mas a análise ainda não foi feita
    elif not st.session_state.analysis_complete:
        st.success("Questionário Concluído!")
        st.markdown("Obrigado por suas respostas. Suas respostas foram registradas. Clique abaixo para gerar a análise.")
        
        if st.button("Gerar Análise (para o Psicólogo)", type="primary"):
            if llm:
                with st.spinner("Analisando respostas..."):
                    try:
                        # Formata as respostas para enviar ao LLM
                        respostas_formatadas = ""
                        for i, q_data in enumerate(QUESTIONARIO):
                            pergunta = q_data["pergunta"]
                            resposta = st.session_state.answers.get(pergunta, "Não respondida")
                            respostas_formatadas += f"**{pergunta}**\n* Resposta: {resposta}\n\n"
                        
                        # Cria o prompt final
                        prompt_final = PROMPT_ANALISE_FINAL.format(
                            RESPOSTAS_DO_USUARIO=respostas_formatadas
                        )
                        
                        # Chama o LLM
                        response_text = llm.invoke(prompt_final)
                        
                        # Salva a análise e marca como completa
                        st.session_state.final_analysis = response_text
                        st.session_state.analysis_complete = True
                        st.rerun() # Recarrega para mostrar a análise
                        
                    except Exception as e:
                        st.error(f"Erro durante a análise: {e}")
            else:
                st.error("O modelo de IA não está disponível para análise.")

    # Se a análise já foi feita, apenas mostre
    else:
        st.subheader("Resultado da Análise de Triagem")
        st.info("Abaixo está um resumo gerado pela IA para revisão do profissional de saúde.")
        
        # Exibe a análise formatada em markdown
        st.markdown(st.session_state.final_analysis)
        
        st.divider()
        if st.button("Reiniciar Questionário"):
            # Limpa o estado para recomeçar
            st.session_state.current_question = 0
            st.session_state.answers = {}
            st.session_state.analysis_complete = False
            st.session_state.pop('final_analysis', None)
            st.rerun()