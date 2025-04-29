import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from models import ProjectSalesPredictor, MonteCarloSimulator
from PIL import Image
import time
from sklearn.model_selection import train_test_split


if 'predictions' not in st.session_state:
    st.session_state.predictions = {
        'new_project': None,
        'ongoing_project': None
    }
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = {
        'new_project': None,
        'ongoing_project': None
    }
if 'training_results' not in st.session_state:
    st.session_state.training_results = {
        'train_metrics_df': None,
        'test_metrics_df': None,
        'test_results': None,
        'fig': None,
        'train_df': None,
        'test_df': None
    }
if 'df' not in st.session_state:
    st.session_state.df = None
if 'project_data' not in st.session_state:
    st.session_state.project_data = {
        'new_project': None,
        'ongoing_project': None
    }
if 'existing_companies' not in st.session_state:
    st.session_state.existing_companies = []


st.set_page_config(
    page_title="Прогнозирование продаж недвижимости",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_models():
    predictor = ProjectSalesPredictor()
    simulator = MonteCarloSimulator(predictor)
    return predictor, simulator

predictor, simulator = load_models()


st.sidebar.title("Навигация")
app_mode = st.sidebar.radio("Выберите режим", 
    ["Обучение модели", "Прогноз для нового проекта", "Анализ текущего проекта"])

def load_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        date_cols = ['start_date', 'end_date', 'agreement_date', 'record_date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        

        st.session_state.existing_companies = df['cons_company'].unique().tolist()
        st.session_state.existing_companies = [str(c) for c in st.session_state.existing_companies if pd.notna(c)]
        
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки файла: {str(e)}")
        return None

def get_company_input(label, default_value=""):
   
    company_options = st.session_state.existing_companies + ["Другая компания..."]
    selected_company = st.selectbox(label, company_options, index=len(company_options)-1 if default_value not in company_options else company_options.index(default_value))
    
    if selected_company == "Другая компания...":
        return st.text_input("Введите название новой строительной компании", value=default_value if default_value not in company_options else "")
    else:
        return selected_company


if app_mode == "Обучение модели":
    st.title("🚀 Обучение модели прогнозирования продаж")
    
    try:
        image = Image.open("image.jpeg")
        st.image(image, caption="Обучите модель прогнозирования на исторических данных", use_column_width=800)
    except:
        st.warning("Не удалось загрузить изображение - продолжаем без него")
    
    st.markdown("""
        <style>
        .training-instructions {
            background-color: #003d1c;
            color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .training-instructions code {
            background-color: #005a2b;
            color: #ffffff;
            padding: 2px 5px;
            border-radius: 3px;
        }
        </style>
        
        <div class="training-instructions">
        <h3>📋 Загрузите данные для обучения</h3>
        <p>Excel-файл должен содержать следующие столбцы (как минимум):</p>
        <ul>
            <li><code>project_name</code>, <code>start_date</code>, <code>end_date</code>, <code>agreement_date</code>, <code>record_date</code></li>
            <li><code>total_area</code>, <code>area</code> (или <code>portion_sold</code>), <code>cons_company</code></li>
        </ul>
        <p>Для лучших результатов используйте данные как минимум по 20 разным проектам.</p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Выберите Excel-файл", type=["xlsx"], key="train_uploader")
    
    if uploaded_file is not None:
        st.session_state.df = load_data(uploaded_file)
        
        if st.session_state.df is not None:
            st.success("✅ Данные успешно загружены!")
            
            with st.expander("🔍 Предпросмотр и сводка данных"):
                st.write("Первые 5 строк данных:")
                st.dataframe(st.session_state.df.head())
                
                st.write("Сводка данных:")
                st.write(f"- Количество проектов: {st.session_state.df['project_name'].nunique()}")
                st.write(f"- Диапазон дат: {st.session_state.df['record_date'].min().date()} до {st.session_state.df['record_date'].max().date()}")
                st.write(f"- Диапазон площадей: {st.session_state.df['total_area'].min():,.0f} до {st.session_state.df['total_area'].max():,.0f} м²")
                st.write(f"- Строительные компании: {len(st.session_state.existing_companies)} уникальных")
            
            if st.button("🚀 Обучить модель", type="primary"):
                with st.spinner("Обучение модели... Это может занять несколько минут"):
                    try:
                        progress_bar = st.progress(0)
                        
                        for percent_complete in range(100):
                            time.sleep(0.02)
                            progress_bar.progress(percent_complete + 1)
                        
                        projects = st.session_state.df['project_name'].unique()
                        train_projects, test_projects = train_test_split(projects, test_size=0.2, random_state=42)
                        train_df = st.session_state.df[st.session_state.df['project_name'].isin(train_projects)]
                        test_df = st.session_state.df[st.session_state.df['project_name'].isin(test_projects)]
                        
                        predictor.fit(train_df)
                        
                        train_metrics = []
                        test_metrics = []
                        
                        for project in train_projects:
                            project_data = train_df[train_df['project_name'] == project]
                            metrics = predictor.evaluate_project(project_data)
                            train_metrics.append(metrics)
                        
                        for project in test_projects:
                            project_data = test_df[test_df['project_name'] == project]
                            metrics = predictor.evaluate_project(project_data)
                            test_metrics.append(metrics)
                        
                        train_metrics_df = pd.DataFrame(train_metrics)
                        test_metrics_df = pd.DataFrame(test_metrics)
                        test_results, fig = predictor.evaluate_and_plot(test_df, num_projects=3)
                        
                        st.session_state.training_results = {
                            'train_metrics_df': train_metrics_df,
                            'test_metrics_df': test_metrics_df,
                            'test_results': test_results,
                            'fig': fig,
                            'train_df': train_df,
                            'test_df': test_df
                        }
                        
                        st.success("🎉 Модель успешно обучена!")
                        
                    except Exception as e:
                        st.error(f"❌ Ошибка при обучении модели: {str(e)}")
    
    if st.session_state.training_results and st.session_state.training_results['train_metrics_df'] is not None:
        train_metrics_df = st.session_state.training_results['train_metrics_df']
        test_metrics_df = st.session_state.training_results['test_metrics_df']
        test_results = st.session_state.training_results['test_results']
        fig = st.session_state.training_results['fig']
        train_df = st.session_state.training_results['train_df']
        test_df = st.session_state.training_results['test_df']
        
        st.subheader("=== Метрики эффективности модели ===")
        
        st.markdown("#### 1. Прогноз периода первой продажи:")
        first_sale_metrics = pd.DataFrame({
            'Dataset': ['Train', 'Test'],
            'MAE': [
                train_metrics_df['first_period_error'].abs().mean(),
                test_metrics_df['first_period_error'].abs().mean()
            ],
            'MSE': [
                (train_metrics_df['first_period_error']**2).mean(),
                (test_metrics_df['first_period_error']**2).mean()
            ]
        })
        st.dataframe(first_sale_metrics.style.format({'MAE': '{:.6f}', 'MSE': '{:.6f}'}))
        
        st.markdown("#### 2. Прогноз периода последней продажи:")
        last_sale_metrics = pd.DataFrame({
            'Dataset': ['Train', 'Test'],
            'MAE': [
                train_metrics_df['last_period_error'].abs().mean(),
                test_metrics_df['last_period_error'].abs().mean()
            ],
            'MSE': [
                (train_metrics_df['last_period_error']**2).mean(),
                (test_metrics_df['last_period_error']**2).mean()
            ]
        })
        st.dataframe(last_sale_metrics.style.format({'MAE': '{:.6f}', 'MSE': '{:.6f}'}))
        
        st.markdown("#### 3. Прогноз объёма продаж:")
        sales_metrics = pd.DataFrame({
            'Dataset': ['Train', 'Test'],
            'MAE (m²)': [
                train_metrics_df['mae'].mean(),
                test_metrics_df['mae'].mean()
            ],
            'MAPE (%)': [
                (train_metrics_df['mae'] / train_df['total_area']).mean() * 100,
                (test_metrics_df['mae'] / test_df['total_area']).mean() * 100
            ],
            'RMSE (m²)': [
                np.sqrt((train_metrics_df['mae']**2).mean()),
                np.sqrt((test_metrics_df['mae']**2).mean())
            ]
        })
        st.dataframe(sales_metrics.style.format({
            'MAE (m²)': '{:.1f}',
            'MAPE (%)': '{:.1f}',
            'RMSE (m²)': '{:.1f}'
        }))
        
        st.subheader("Результаты оценки модели")
        tab1, tab2 = st.tabs(["📈 Графики производительности", "📊 Таблица метрик"])
        
        with tab1:
            st.markdown("### Производительность модели на тестовых проектах")
            st.pyplot(fig)
            
        with tab2:
            st.markdown("### Детальные метрики оценки")
            st.dataframe(
                test_results.style.format({
                    'mae': '{:.1f}',
                    'r2_score': '{:.2f}',
                    'first_period_error': '{:.3f}',
                    'last_period_error': '{:.3f}'
                }),
                use_container_width=True
            )


elif app_mode == "Прогноз для нового проекта":
    st.title("🔮 Прогноз продаж для нового проекта")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Детали проекта")
        project_name = st.text_input("Название проекта", "Яссауи")
        start_date = st.date_input("Дата начала", datetime(2022, 11, 30))
        end_date = st.date_input("Дата окончания", datetime(2024, 12, 30))
        agreement_date = st.date_input("Дата соглашения", datetime(2023, 2, 2))
        total_area = st.number_input("Общая площадь (м²)", min_value=100, value=10795)
        duration = st.number_input("Продолжительность (месяцы)", min_value=1, value=25)
        company = get_company_input("Строительная компания", "ТОО BAZIS-A Corp.")
    
    with col2:
        st.subheader("Параметры цены")
        min_price = st.number_input("Минимальная цена за м²", min_value=100, value=450000)
        max_price = st.number_input("Максимальная цена за м²", min_value=100, value=560000)
        
        st.subheader("Монте-Карло симуляция")
        cost_official = st.number_input("Общая стоимость проекта", min_value=0, value=6548335303)
        unfin_cost = st.number_input("Стоимость незавершенки", min_value=0, value=389462312)
    
    if st.button("Сформировать прогноз"):
        project_data = {
            "project_name": project_name,
            "start_date": start_date.strftime('%Y-%m-%d'),
            "end_date": end_date.strftime('%Y-%m-%d'),
            "agreement_date": agreement_date.strftime('%Y-%m-%d'),
            "total_area": total_area,
            "duration": duration,
            "cons_company": company
        }
        
        st.session_state.project_data['new_project'] = project_data
        
        with st.spinner("Формирование прогноза..."):
            try:
                predictions = predictor.predict(project_data, min_price, max_price)
                st.session_state.predictions['new_project'] = predictions
                
                simulation_results = simulator.run_simulation(
                    project_data=project_data,
                    min_price=min_price,
                    max_price=max_price,
                    cost_official=cost_official,
                    unfin_cost=unfin_cost,
                    num_simulations=10000
                )
                st.session_state.simulation_results['new_project'] = simulation_results
                
                st.success("Прогноз успешно сформирован!")
                
            except Exception as e:
                st.error(f"Ошибка при формировании прогноза: {str(e)}")
    
    if st.session_state.predictions['new_project'] and st.session_state.project_data['new_project']:
        predictions = st.session_state.predictions['new_project']
        simulation_results = st.session_state.simulation_results['new_project']
        project_data = st.session_state.project_data['new_project']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Первый месяц продаж", f"{predictions['first_sale_period']:.1%}")
            st.metric("Последний месяц продаж", f"{predictions['last_sale_period']:.1%}")
            st.metric("Продолжительность продаж (дни)", f"{predictions['sales_duration_days']:.0f}")
        
        with col2:
            st.metric("Прогнозируемая выручка", f"${predictions['total_predicted_revenue']:,.0f}")
            st.metric("Средние продажи в месяц", f"{predictions['avg_monthly_sales']:,.1f} м²")
            st.metric("Месяцев с продажами", len(predictions['monthly_predictions']))
        
        st.subheader("Помесячный прогноз продаж")
        st.dataframe(predictions['monthly_predictions'])
        
        st.subheader("График прогноза продаж")
        forecast_fig = predictor.plot_sales_forecast(predictions, project_data)
        st.pyplot(forecast_fig)
        
        st.subheader("Симуляция ценового диапазона")
        price_fig = simulator.plot_price_diapason(predictions['monthly_predictions'], project_name)
        st.pyplot(price_fig)
        
        st.subheader("Результаты Монте-Карло симуляции")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Вероятность неудачи", f"{simulation_results['failure_rate']:.1%}")
            st.metric("Средний денежный поток", f"${simulation_results['mean_cash_flow']:,.0f}")
            st.metric("Стандартное отклонение", f"${simulation_results['std_dev_cash_flow']:,.0f}")
        
        with col2:
            st.metric("Лучший сценарий", f"${simulation_results['max_cash_flow']:,.0f}")
            st.metric("Худший сценарий", f"${simulation_results['min_cash_flow']:,.0f}")
            st.metric("Медианное время окупаемости", simulation_results['median_months_to_breakeven'])
        
        st.subheader("Распределение симуляции")
        sim_fig = simulator._plot_monte_carlo_results(
            simulation_results['simulation_results'],
            simulation_results['mean_cash_flow'],
            simulation_results['std_dev_cash_flow'],
            simulation_results['failure_rate'],
            project_name,
            cost_official,
            unfin_cost
        )
        st.pyplot(sim_fig)


elif app_mode == "Анализ текущего проекта":
    st.title("🔮 Прогноз продаж для текущего проекта")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Детали проекта")
        project_name = st.text_input("Название проекта", "4YOU Business - 1", key="ongoing_name")
        start_date = st.date_input("Дата начала", datetime(2019, 9, 10), key="ongoing_start")
        end_date = st.date_input("Дата окончания", datetime(2022, 1, 10), key="ongoing_end")
        agreement_date = st.date_input("Дата соглашения", datetime(2019, 10, 10), key="ongoing_agree")
        total_area = st.number_input("Общая площадь (м²)", min_value=100, value=19824, key="ongoing_area")
        duration = st.number_input("Продолжительность (месяцы)", min_value=1, value=28, key="ongoing_duration")
        company = get_company_input("Строительная компания", "ТОО Мереке Сервис НС")
    
    with col2:
        st.subheader("Исторические данные продаж")
        completed_months = st.number_input("Завершенных месяцев", min_value=1, max_value=120, value=8)
        actual_sales = st.text_input("Фактические продажи (через запятую)", "1360.85007, 2517.27002, 1200.700005, 734.4100037, 998.2700043, 946.3999977, 274.9700012, 1344.429989")
        period_percentages = st.text_input("Процент завершения (через запятую)", "0.037514654, 0.060961313, 0.103165299, 0.148886284, 0.168815944, 0.203985932, 0.297772567, 0.385697538")
        
        st.subheader("Параметры цены")
        min_price = st.number_input("Минимальная цена за м²", min_value=100, value=501000, key="ongoing_min")
        max_price = st.number_input("Максимальная цена за м²", min_value=100, value=580000, key="ongoing_max")
        
        st.subheader("Монте-Карло симуляция")
        cost_official = st.number_input("Общая стоимость проекта", min_value=0, value=12457003488, key="ongoing_cost")
        unfin_cost = st.number_input("Стоимость незавершенки", min_value=0, value=47120000, key="ongoing_unfin")
    
    if st.button("Обновить прогноз"):
        project_data = {
            "project_name": project_name,
            "start_date": start_date.strftime('%Y-%m-%d'),
            "end_date": end_date.strftime('%Y-%m-%d'),
            "agreement_date": agreement_date.strftime('%Y-%m-%d'),
            "total_area": total_area,
            "duration": duration,
            "cons_company": company
        }
        
        st.session_state.project_data['ongoing_project'] = project_data
        
        with st.spinner("Обновление прогноза..."):
            try:
                actual_sales_list = [float(x.strip()) for x in actual_sales.split(",")] if actual_sales else []
                period_pct_list = [float(x.strip()) for x in period_percentages.split(",")] if period_percentages else []
                
                predictions = predictor.predict_ongoing_project(
                    project_data=project_data,
                    completed_months=completed_months,
                    actual_sales=actual_sales_list,
                    period_percentages=period_pct_list
                )
                
                if 'price_per_sqm' not in predictions['monthly_predictions']:
                    predictions['monthly_predictions']['price_per_sqm'] = np.linspace(
                        min_price, max_price, len(predictions['monthly_predictions'])
                    )
                
                st.session_state.predictions['ongoing_project'] = predictions
                
                simulation_results = simulator.run_simulation(
                    project_data=project_data,
                    min_price=min_price,
                    max_price=max_price,
                    cost_official=cost_official,
                    unfin_cost=unfin_cost,
                    num_simulations=10000
                )
                st.session_state.simulation_results['ongoing_project'] = simulation_results
                
                st.success("Прогноз успешно обновлен!")
                
            except Exception as e:
                st.error(f"Ошибка при обновлении прогноза: {str(e)}")
    
    if st.session_state.predictions['ongoing_project'] and st.session_state.project_data['ongoing_project']:
        predictions = st.session_state.predictions['ongoing_project']
        simulation_results = st.session_state.simulation_results['ongoing_project']
        project_data = st.session_state.project_data['ongoing_project']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Продано на данный момент", f"{predictions['total_sold_so_far']:,.0f} м²")
            st.metric("Осталось продать", f"{predictions['remaining_area']:,.0f} м²")
            st.metric("Процент завершения", f"{predictions['completion_percentage']:.1%}")
        
        with col2:
            st.metric("Последний месяц продаж", f"{predictions['last_sale_period']:.1%}")
            st.metric("Осталось дней продаж", f"{predictions['sales_duration_days']:.0f}")
            st.metric("Осталось месяцев с продажами", len(predictions['monthly_predictions']))
        
        st.subheader("Помесячный прогноз продаж")
        st.dataframe(predictions['monthly_predictions'])
        
        st.subheader("Обновленный график прогноза")
        ongoing_fig = predictor.plot_ongoing_project_results(predictions, project_data)
        st.pyplot(ongoing_fig)
        
        st.subheader("Симуляция ценового диапазона")
        price_fig = simulator.plot_ongoing_project_diapason(
            predictions, 
            project_data,
            min_price,
            max_price
        )
        st.pyplot(price_fig)
        
        st.subheader("Результаты Монте-Карло симуляции")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Вероятность неудачи", f"{simulation_results['failure_rate']:.1%}")
            st.metric("Средний денежный поток", f"${simulation_results['mean_cash_flow']:,.0f}")
            st.metric("Стандартное отклонение", f"${simulation_results['std_dev_cash_flow']:,.0f}")
        
        with col2:
            st.metric("Лучший сценарий", f"${simulation_results['max_cash_flow']:,.0f}")
            st.metric("Худший сценарий", f"${simulation_results['min_cash_flow']:,.0f}")
            st.metric("Медианное время окупаемости", simulation_results['median_months_to_breakeven'])
        
        st.subheader("Распределение симуляции")
        sim_fig = simulator._plot_monte_carlo_results(
            simulation_results['simulation_results'],
            simulation_results['mean_cash_flow'],
            simulation_results['std_dev_cash_flow'],
            simulation_results['failure_rate'],
            project_name,
            cost_official,
            unfin_cost
        )
        st.pyplot(sim_fig)