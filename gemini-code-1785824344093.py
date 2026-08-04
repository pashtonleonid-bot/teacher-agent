import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from docx import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(
    page_title="Педагогический Ассистент (Биология / Химия)",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 ИИ-Агент: Педагогический Ассистент Учителя")
st.caption("Автоматический анализ контрольных материалов по Блуму и проверка работ учеников.")

# Функция для извлечения текста из разных источников
def extract_text(file_upload, url_input):
    if file_upload is not None:
        file_type = file_upload.name.split('.')[-1].lower()
        if file_type == 'pdf':
            reader = PdfReader(file_upload)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        elif file_type in ['docx', 'doc']:
            doc = Document(file_upload)
            return "\n".join([p.text for p in doc.paragraphs])
        elif file_type == 'txt':
            return file_upload.getvalue().decode("utf-8")
    elif url_input:
        try:
            response = requests.get(url_input, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            # Удаляем скрипты и стили
            for script in soup(["script", "style"]):
                script.extract()
            return soup.get_text(separator=' ', strip=True)
        except Exception as e:
            st.error(f"Ошибка при чтении ссылки: {e}")
            return None
    return None

with st.sidebar:
    st.header("⚙️ Параметры проверки")
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.environ.get("OPENAI_API_KEY", ""),
        help="Введите ваш ключ API"
    )
    
    subject = st.selectbox("Предмет", ["Биология", "Химия"])
    grade_level = st.selectbox("Класс", ["7 класс", "8 класс", "9 класс", "10 класс", "11 класс"])
    assessment_type = st.radio(
        "Тип оценивания",
        ["Формирующее (развивающая обратная связь)", "Констатирующее (итоговая оценка и подведение итогов)"]
    )

st.markdown("---")

# 1️⃣ ШАГ 1: Загрузка материалов урока и анализ по Таксономии Блума
st.subheader("1️⃣ Загрузка заданий / контрольной работы")
col_task_file, col_task_url = st.columns(2)

with col_task_file:
    task_file = st.file_uploader("Загрузить файл с заданиями (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], key="task_file")
with col_task_url:
    task_url = st.text_input("Или вставьте ссылку на задание", placeholder="https://...", key="task_url")

# 2️⃣ ШАГ 2: Загрузка работы ученика
st.subheader("2️⃣ Загрузка работы ученика")
col_stud_file, col_stud_url = st.columns(2)

with col_stud_file:
    student_file = st.file_uploader("Загрузить файл с ответом ученика (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], key="student_file")
with col_stud_url:
    student_url = st.text_input("Или вставьте ссылку на ответ ученика", placeholder="https://...", key="student_url")

st.markdown("---")

# Промпт для ИИ
SYSTEM_PROMPT = """
Ты — профессиональный, заботливый учитель {subject} для {grade_level}.
Твоя задача — провести детальный двухэтапный анализ работы.

Параметры:
- Предмет: {subject} | Класс: {grade_level}
- Тип оценивания: {assessment_type}

ВЫПОЛНИ АНАЛИЗ ПО СЛЕДУЮЩЕЙ СТРУКТУРЕ:

### 📊 Раздел 1. Картирование заданий по Таксономии Блума
Разбери контрольный материал по номерам заданий и определи, какой уровень Блума проверяет каждое из них:
- № Задания: Уровень Блума (Знание / Понимание / Применение / Анализ / Оценка / Создание) — краткое обоснование.

---

### 💬 Раздел 2. Обратная связь по работе ученика
Сформируй развернутый комментарий ученику по структуре:
1. 🌟 **Что сделано отлично:** Укажи конкретные номера заданий и верные логические шаги/термины.
2. 🎯 **Где допущена ошибка / неточность:** Разбери ошибки с привязкой к номерам заданий и категориям Блума (например, "в №3 на уровне Применение...").
3. ❓ **Направляющий вопрос / рекомендация:**
   - Если формирующее оценивание: Задай наводящие вопросы к ошибочным заданиям (без готового ответа).
   - Если констатирующее оценивание: Дай четкие рекомендации по повторению тем и подведи итог.

Тон ответа: поддерживающий, академический, уважительный.
ВАЖНО: Весь текст должен быть выровнен справа налево (Right-to-Left).
"""

# 3️⃣ ШАГ 3: Запуск анализа
if st.button("🚀 Запустить анализ работы", type="primary", use_container_width=True):
    if not api_key_input:
        st.error("Пожалуйста, введите OpenAI API Key в боковой панели!")
        st.stop()
        
    task_text = extract_text(task_file, task_url)
    student_text = extract_text(student_file, student_url)
    
    if not task_text:
        st.warning("Загрузите файл с заданиями или укажите ссылку!")
        st.stop()
        
    if not student_text:
        st.warning("Загрузите файл с ответом ученика или укажите ссылку!")
        st.stop()
        
    os.environ["OPENAI_API_KEY"] = api_key_input
    
    with st.spinner("Анализируем задания по Блуму и проверяем работу ученика..."):
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
            prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human", "Условие заданий:\n{task}\n\nОтветы ученика:\n{student_answer}")
            ])
            
            chain = prompt | llm
            
            response = chain.invoke({
                "subject": subject,
                "grade_level": grade_level,
                "assessment_type": assessment_type,
                "task": task_text,
                "student_answer": student_text
            })
            
            st.markdown("## 📈 Результаты анализа")
            st.success(response.content)
            
        except Exception as e:
            st.error(f"Произошла ошибка при анализе: {e}")
