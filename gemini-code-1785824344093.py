import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(
    page_title="Педагогический Ассистент (Биология / Химия)",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 ИИ-Агент: Педагогический Ассистент Учителя")
st.caption("Формирующее и констатирующее оценивание работ по биологии и химии на основе Таксономии Блума.")

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

st.subheader("📝 Загрузка материалов урока и работы ученика")

col1, col2 = st.columns(2)

with col1:
    task_description = st.text_area(
        "Задание / Критерии оценивания",
        height=200,
        placeholder="Введите условие задачи или вопросы..."
    )

with col2:
    student_response = st.text_area(
        "Ответ ученика",
        height=200,
        placeholder="Вставьте ответ ученика..."
    )

SYSTEM_PROMPT = """
Ты — профессиональный, заботливый и высококвалифицированный учитель {subject} для учеников {grade_level}.
Твоя цель — проанализировать работу ученика и дать обратную связь.

Параметры проверки:
- Предмет: {subject}
- Класс: {grade_level}
- Тип оценивания: {assessment_type}

Контекст анализа (Таксономия Блума):
Определи, на каком когнитивном уровне (Знание, Понимание, Применение, Анализ, Оценка, Создание) выстроено задание и где справился или допустил ошибку ученик.

Правила формирования комментария ученику:
1. 🌟 Что сделано отлично: Укажи конкретные успехи, верные термины или правильные шаги рассуждения.
2. 🎯 Где допущена ошибка / неточность: Бережно и точно покажи неточности или логические пробелы.
3. ❓ Направляющий вопрос или рекомендация: 
   - Для формирующего оценивания: Дай наводящий вопрос (без прямого готового ответа), помогающий ученику самому додумать решение.
   - Для констатирующего оценивания: Дай конкретную рекомендацию по повторению темы и подведи итог выполнения критериев.

Тон ответа: поддерживающий, академический, уважительный и адаптированный под возраст учеников {grade_level}.
Текст должен быть выровнен справа налево (Right-to-Left) для поддержки корректной ориентации верстки при представлении.
"""

if st.button("Сгенерировать обратную связь", type="primary"):
    if not api_key_input:
        st.error("Пожалуйста, введите OpenAI API Key в боковой панели!")
        st.stop()
        
    if not task_description or not student_response:
        st.warning("Заполните оба поля: условие задания и ответ ученика!")
        st.stop()
        
    os.environ["OPENAI_API_KEY"] = api_key_input
    
    with st.spinner("Анализ работы и формирование комментария..."):
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
            prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human", "Условие задания:\n{task}\n\nОтвет ученика:\n{student_answer}")
            ])
            
            chain = prompt | llm
            
            response = chain.invoke({
                "subject": subject,
                "grade_level": grade_level,
                "assessment_type": assessment_type,
                "task": task_description,
                "student_answer": student_response
            })
            
            st.markdown("### 💬 Готовый комментарий для ученика:")
            st.success(response.content)
            
        except Exception as e:
            st.error(f"Произошла ошибка при генерации: {e}")