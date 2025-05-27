import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Аутентификация
def check_password():
    """Возвращает `True`, если пользователь ввел правильный пароль."""
    def password_entered():
        if st.session_state.get("username") == "vokko" and st.session_state.get("password") == "vokko":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        username = st.text_input("Логин", key="username", on_change=password_entered)
        password = st.text_input("Пароль", type="password", key="password", on_change=password_entered)
        st.button("Войти", on_click=password_entered)
        return False
    elif not st.session_state["password_correct"]:
        username = st.text_input("Логин", key="username", on_change=password_entered)
        password = st.text_input("Пароль", type="password", key="password", on_change=password_entered)
        st.button("Войти", on_click=password_entered)
        st.error("Неверный логин или пароль")
        return False
    else:
        return True

if not check_password():
    st.stop()

# Настройка страницы
st.set_page_config(
    page_title="Визуализация и сравнение данных по зонам",
    layout="wide"
)

# Цветовая палитра
COLOR_PALETTE = [
    "#4285F4", "#EA4335", "#FBBC05", "#34A853", "#FF6D01",
    "#46BFBD", "#F7464A", "#949FB1", "#9C27B0", "#673AB7",
    "#3F51B5", "#2196F3", "#03A9F4", "#00BCD4", "#009688",
    "#4CAF50", "#8BC34A", "#CDDC39", "#FFEB3B", "#FFC107"
]

# Группы зон
AREA_GROUPS = {
    "Алматинка север": ["Зона 4", "Зона 5", "Зона 6", "Зона 7"],
    "Алматинка юг": ["Зона 10", "Зона 11", "Зона 12", "Зона 13"],
    "Анкара": ["Зона 8", "Зона 9"],
    "Горький": ["Зона 1", "Зона 2", "Зона 3"],
}

# Названия дней недели
WEEKDAYS = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]

# Заголовок
st.title("Визуализация и сравнение данных по зонам")

@st.cache_data(ttl=300)
def load_google_sheet():
    try:
        scope = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/drive.file"
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_name("vokko-projects-df95fd45f19c.json", scope)
        client = gspread.authorize(creds)
        spreadsheet = client.open_by_key("1oloDZ2C-KK2vMcuUm-h6oto5ZythG954fe0AhxW8sKQ")
        worksheet = spreadsheet.get_worksheet(0)
        data = worksheet.get_all_values()
        df = pd.DataFrame(data[1:], columns=data[0])
        required_columns = ['Часы'] + [f'Зона {i}' for i in range(1, 14)]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Отсутствуют обязательные колонки: {', '.join(missing_columns)}")
            return None
        return df[required_columns]
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {str(e)}")
        return None

df = load_google_sheet()
if df is None:
    st.stop()

zone_columns = [col for col in df.columns if col.startswith("Зона")]
available_dates = df["Часы"].str.split(" ").str[0].unique()

# Parse dates to datetime objects for proper sorting
def parse_date(date_str):
    try:
        day, month, year = date_str.split('.')
        return datetime(int(year), int(month), int(day))
    except:
        return datetime.min

available_dates = sorted(available_dates, key=parse_date)

date_options = []
for date_str in available_dates:
    try:
        date_parts = date_str.split(".")
        if len(date_parts) == 3:
            date_obj = datetime.strptime(f"{date_parts[2]}-{date_parts[1]}-{date_parts[0]}", "%Y-%m-%d")
            weekday = WEEKDAYS[date_obj.weekday()]
            date_options.append((date_str, weekday))
    except:
        date_options.append((date_str, ""))

st.session_state.df = df
st.session_state.zone_columns = zone_columns
st.session_state.date_options = date_options

st.success("Данные успешно загружены из Google Sheets")

# Выбор режима
st.subheader("Режим отображения:")
mode_col1, mode_col2 = st.columns(2)

with mode_col1:
    display_mode = st.radio(
        "Тип отображения:",
        ["Одна дата", "Сравнение двух дат"],
        key="display_mode"
    )

with mode_col2:
    view_mode = st.radio(
        "Группировка:",
        ["Отдельные зоны", "Группы зон (улицы)"],
        key="view_mode"
    )

# Выбор дат
if display_mode == "Одна дата":
    selected_date = st.selectbox(
        "Выберите дату:",
        options=[f"{date} ({weekday})" if weekday else date for date, weekday in st.session_state.date_options],
        key="single_date"
    )
    selected_date = selected_date.split(" ")[0]
else:
    date_col1, date_col2 = st.columns(2)
    with date_col1:
        date1 = st.selectbox(
            "Первая дата:",
            options=[f"{date} ({weekday})" if weekday else date for date, weekday in st.session_state.date_options],
            key="date1"
        )
        date1 = date1.split(" ")[0]
    with date_col2:
        date2 = st.selectbox(
            "Вторая дата:",
            options=[f"{date} ({weekday})" if weekday else date for date, weekday in st.session_state.date_options],
            index=1 if len(st.session_state.date_options) > 1 else 0,
            key="date2"
        )
        date2 = date2.split(" ")[0]

# Выбор зон/групп
if view_mode == "Отдельные зоны":
    selected_items = st.multiselect(
        "Выберите зоны для отображения:",
        options=st.session_state.zone_columns,
        default=st.session_state.zone_columns,
        key="zone_selection"
    )
else:
    selected_items = st.multiselect(
        "Выберите группы зон для отображения:",
        options=list(AREA_GROUPS.keys()),
        default=list(AREA_GROUPS.keys()),
        key="area_selection"
    )

if not selected_items:
    st.warning("Выберите хотя бы одну зону/группу для отображения")
    st.stop()

# Фильтрация данных
def filter_data(date):
    filtered = st.session_state.df[st.session_state.df["Часы"].str.startswith(date)].copy()
    filtered["time"] = filtered["Часы"].str.split(" ").str[1]
    all_hours = pd.DataFrame({"time": [f"{h:02d}:00" for h in range(24)]})
    if not filtered.empty and "time" in filtered.columns:
        filtered["time"] = filtered["time"].apply(
            lambda x: f"{int(x.split(':')[0]):02d}:00" if ":" in str(x) else "00:00"
        )
    filtered = pd.merge(all_hours, filtered, on="time", how="left")
    zone_cols = [col for col in filtered.columns if col.startswith("Зона")]
    for col in zone_cols:
        if col in filtered.columns:
            filtered[col] = pd.to_numeric(filtered[col], errors='coerce').fillna(0)  # Заполняем пропуски нулями
    filtered["time_sort"] = filtered["time"].apply(
        lambda x: int(x.split(":")[0]) if ":" in str(x) else 0
    )
    filtered = filtered.sort_values(by="time_sort")
    filtered = filtered.drop("time_sort", axis=1)
    return filtered

if display_mode == "Одна дата":
    filtered_data = filter_data(selected_date)
else:
    filtered_data1 = filter_data(date1)
    filtered_data2 = filter_data(date2)

# Подготовка данных для графиков
def prepare_chart_data(filtered_data, selected_items, is_area_view=False):
    all_hours = [f"{h:02d}:00" for h in range(24)]
    labels = all_hours
    time_df = pd.DataFrame({"time": all_hours})
    if "time" not in filtered_data.columns:
        filtered_data["time"] = "00:00"
    filtered_data = pd.merge(time_df, filtered_data, on="time", how="left")
    filtered_data["time_sort"] = filtered_data["time"].apply(
        lambda x: int(x.split(":")[0]) if ":" in str(x) else 0
    )
    filtered_data = filtered_data.sort_values(by="time_sort")
    filtered_data = filtered_data.drop("time_sort", axis=1)
    datasets = []
    if is_area_view:
        for i, area in enumerate(selected_items):
            zones = AREA_GROUPS[area]
            available_zones = [zone for zone in zones if zone in filtered_data.columns]
            if not available_zones:
                datasets.append({
                    "name": area,
                    "data": pd.Series([0] * len(filtered_data)),  # Заполняем нулями
                    "color": COLOR_PALETTE[i % len(COLOR_PALETTE)]
                })
                continue
            area_data = filtered_data[available_zones].apply(pd.to_numeric, errors='coerce').mean(axis=1)
            datasets.append({
                "name": area,
                "data": area_data,
                "color": COLOR_PALETTE[i % len(COLOR_PALETTE)]
            })
    else:
        for i, zone in enumerate(selected_items):
            if zone not in filtered_data.columns:
                datasets.append({
                    "name": zone,
                    "data": pd.Series([0] * len(filtered_data)),  # Заполняем нулями
                    "color": COLOR_PALETTE[i % len(COLOR_PALETTE)]
                })
                continue
            zone_data = pd.to_numeric(filtered_data[zone], errors='coerce').fillna(0)  # Заполняем пропуски нулями
            datasets.append({
                "name": zone,
                "data": zone_data,
                "color": COLOR_PALETTE[i % len(COLOR_PALETTE)]
            })
    return labels, datasets

# Отображение графиков
if display_mode == "Одна дата":
    labels, datasets = prepare_chart_data(
        filtered_data, 
        selected_items, 
        view_mode == "Группы зон (улицы)"
    )
    # Линейный график
    fig = go.Figure()
    for dataset in datasets:
        fig.add_trace(go.Scatter(
            x=labels,
            y=dataset["data"],
            name=dataset["name"],
            line=dict(color=dataset["color"], width=2),
            mode="lines",
            connectgaps=True
        ))
    weekday = next((wd for date, wd in st.session_state.date_options if date == selected_date), "")
    fig.update_layout(
        title=f"Данные {'по группам зон' if view_mode == 'Группы зон (улицы)' else 'по зонам'} за {selected_date} {f'({weekday})' if weekday else ''}",
        xaxis_title="Время",
        yaxis_title="Значение",
        height=500,
        hovermode="x unified"
    )
    fig.update_xaxes(
        tickmode='array',
        tickvals=[f"{h:02d}:00" for h in range(24)],
        ticktext=[f"{h:02d}:00" for h in range(0, 24, 1)],
        tickangle=45,
        categoryorder='array',
        categoryarray=[f"{h:02d}:00" for h in range(24)]
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Эффективность (столбчатый график)
    if view_mode == "Группы зон (улицы)":
        st.subheader("Эффективность по группам зон:")
        efficiency_data = []
        for area in selected_items:
            zones = AREA_GROUPS[area]
            area_data = filtered_data[zones].apply(pd.to_numeric, errors='coerce').mean(axis=1)
            avg_value = area_data.mean()
            max_value = area_data.max()
            efficiency = (avg_value / max_value) * 100 if max_value != 0 else 0
            efficiency_data.append({
                "area": area,
                "efficiency": efficiency
            })
        
        eff_df = pd.DataFrame(efficiency_data)
        eff_df["efficiency"] = eff_df["efficiency"].round(1)
        
        # Создание столбчатого графика
        fig_eff = go.Figure()
        for i, row in eff_df.iterrows():
            fig_eff.add_trace(go.Bar(
                x=[row["area"]],
                y=[row["efficiency"]],
                name=row["area"],
                marker_color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
                width=0.4,
                showlegend=True
            ))
            fig_eff.add_annotation(
                x=row["area"],
                y=row["efficiency"] + 5,
                text=f"{row['efficiency']:.1f}%",
                showarrow=False,
                font=dict(size=12)
            )
        
        fig_eff.update_layout(
            title=f"Эффективность по группам зон за {selected_date} {f'({weekday})' if weekday else ''}",
            yaxis_title="Эффективность (%)",
            yaxis_range=[0, 100],
            height=400
        )
        st.plotly_chart(fig_eff, use_container_width=True)
        
        # Отображение таблицы эффективности
        def color_efficiency(val):
            color = "green" if val >= 50 else "orange" if val >= 30 else "red"
            return f"color: {color}"
        
        st.dataframe(
            eff_df.style.applymap(color_efficiency, subset=["efficiency"]),
            column_config={
                "area": "Группа зон",
                "efficiency": st.column_config.ProgressColumn(
                    "Эффективность (%)",
                    format="%.1f",
                    min_value=0,
                    max_value=100
                )
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Общая эффективность
        all_values = []
        for area in selected_items:
            zones = AREA_GROUPS[area]
            all_values.extend(filtered_data[zones].values.flatten().tolist())
        
        if all_values:
            avg_total = np.mean([x for x in all_values if pd.notna(x)])
            max_total = np.nanmax(all_values)
            efficiency_total = (avg_total / max_total) * 100 if max_total != 0 else 0
            st.metric(
                label=f"Общая эффективность ({selected_date} {f'({weekday})' if weekday else ''})",
                value=f"{efficiency_total:.1f}%"
            )

else:
    labels1, datasets1 = prepare_chart_data(
        filtered_data1, 
        selected_items, 
        view_mode == "Группы зон (улицы)"
    )
    labels2, datasets2 = prepare_chart_data(
        filtered_data2, 
        selected_items, 
        view_mode == "Группы зон (улицы)"
    )
    # Линейный график для сравнения двух дат
    fig = go.Figure()
    for i, (dataset1, dataset2) in enumerate(zip(datasets1, datasets2)):
        if dataset1["name"] != dataset2["name"]:
            st.error(f"Несоответствие имен: {dataset1['name']} != {dataset2['name']}")
            continue
        weekday1 = next((wd for date, wd in st.session_state.date_options if date == date1), "")
        weekday2 = next((wd for date, wd in st.session_state.date_options if date == date2), "")
        fig.add_trace(go.Scatter(
            x=labels1,
            y=dataset1["data"],
            name=f"{dataset1['name']} ({date1} {f'({weekday1})' if weekday1 else ''})",
            line=dict(color=dataset1["color"], width=2),
            mode="lines",
            connectgaps=True
        ))
        fig.add_trace(go.Scatter(
            x=labels2,
            y=dataset2["data"],
            name=f"{dataset2['name']} ({date2} {f'({weekday2})' if weekday2 else ''})",
            line=dict(color=dataset2["color"], width=2, dash="dash"),
            mode="lines",
            connectgaps=True
        ))
    fig.update_layout(
        title=f"Сравнение {'групп зон' if view_mode == 'Группы зон (улицы)' else 'зон'}: {date1} {f'({weekday1})' if weekday1 else ''} и {date2} {f'({weekday2})' if weekday2 else ''}",
        xaxis_title="Время",
        yaxis_title="Значение",
        height=500,
        hovermode="x unified"
    )
    fig.update_xaxes(
        tickmode='array',
        tickvals=[f"{h:02d}:00" for h in range(24)],
        ticktext=[f"{h:02d}:00" for h in range(0, 24, 1)],
        tickangle=45,
        categoryorder='array',
        categoryarray=[f"{h:02d}:00" for h in range(24)]
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Эффективность для сравнения двух дат
    if view_mode == "Группы зон (улицы)":
        efficiency_data = []
        for i, area in enumerate(selected_items):
            zones = AREA_GROUPS[area]
            area_data1 = filtered_data1[zones].apply(pd.to_numeric, errors='coerce').mean(axis=1)
            avg_value1 = area_data1.mean()
            max_value1 = area_data1.max()
            efficiency1 = (avg_value1 / max_value1) * 100 if max_value1 != 0 else 0
            area_data2 = filtered_data2[zones].apply(pd.to_numeric, errors='coerce').mean(axis=1)
            avg_value2 = area_data2.mean()
            max_value2 = area_data2.max()
            efficiency2 = (avg_value2 / max_value2) * 100 if max_value2 != 0 else 0
            difference = efficiency2 - efficiency1
            efficiency_data.append({
                "area": area,
                f"{date1}": efficiency1,
                f"{date2}": efficiency2,
                "difference": difference
            })
        eff_df = pd.DataFrame(efficiency_data)
        fig = go.Figure()
        for i, row in eff_df.iterrows():
            fig.add_trace(go.Bar(
                x=[row["area"]],
                y=[row[date1]],
                name=f"{date1}",
                marker_color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
                width=0.4,
                offset=-0.2,
                showlegend=i == 0
            ))
            fig.add_trace(go.Bar(
                x=[row["area"]],
                y=[row[date2]],
                name=f"{date2}",
                marker_color=px.colors.sequential.algae[i % len(px.colors.sequential.algae)],
                width=0.4,
                offset=0.2,
                showlegend=i == 0
            ))
            diff_text = f"+{row['difference']:.1f}%" if row['difference'] >= 0 else f"{row['difference']:.1f}%"
            diff_color = "green" if row['difference'] >= 0 else "red"
            fig.add_annotation(
                x=row["area"],
                y=max(row[date1], row[date2]) + 5,
                text=diff_text,
                showarrow=False,
                font=dict(color=diff_color, size=12)
            )
        fig.update_layout(
            title=f"Сравнение эффективности: {date1} {f'({weekday1})' if weekday1 else ''} и {date2} {f'({weekday2})' if weekday2 else ''}",
            barmode="group",
            yaxis_title="Эффективность (%)",
            yaxis_range=[0, 100],
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Эффективность по группам зон:")
        eff_display = eff_df.copy()
        eff_display[date1] = eff_display[date1].round(1).astype(str) + "%"
        eff_display[date2] = eff_display[date2].round(1).astype(str) + "%"
        eff_display["Изменение"] = eff_display["difference"].apply(
            lambda x: f"+{x:.1f}%" if x >= 0 else f"{x:.1f}%"
        )
        def color_diff(val):
            try:
                num = float(val.replace("%", "").replace("+", ""))
                color = "green" if num >= 0 else "red"
                return f"color: {color}"
            except:
                return ""
        styled_df = eff_display[["area", date1, date2, "Изменение"]].reset_index(drop=True)
        st.dataframe(
            styled_df.style.map(
                color_diff, subset=["Изменение"]
            ),
            column_config={
                "area": "Группа зон",
                date1: date1,
                date2: date2,
                "Изменение": "Изменение"
            },
            use_container_width=True,
            hide_index=True
        )
        all_values1 = []
        all_values2 = []
        for area in selected_items:
            zones = AREA_GROUPS[area]
            all_values1.extend(pd.to_numeric(filtered_data1[zones].values.flatten(), errors='coerce').tolist())
            all_values2.extend(pd.to_numeric(filtered_data2[zones].values.flatten(), errors='coerce').tolist())
        if all_values1 and all_values2:
            avg_total1 = np.mean([x for x in all_values1 if pd.notna(x)])
            max_total1 = np.nanmax(all_values1)
            efficiency_total1 = (avg_total1 / max_total1) * 100 if max_total1 != 0 else 0
            avg_total2 = np.mean([x for x in all_values2 if pd.notna(x)])
            max_total2 = np.nanmax(all_values2)
            efficiency_total2 = (avg_total2 / max_total2) * 100 if max_total2 != 0 else 0
            total_diff = efficiency_total2 - efficiency_total1
            diff_text = f"+{total_diff:.1f}%" if total_diff >= 0 else f"{total_diff:.1f}%"
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label=f"Общая эффективность ({date1} {f'({weekday1})' if weekday1 else ''})",
                    value=f"{efficiency_total1:.1f}%"
                )
            with col2:
                st.metric(
                    label=f"Общая эффективность ({date2} {f'({weekday2})' if weekday2 else ''})",
                    value=f"{efficiency_total2:.1f}%"
                )
            with col3:
                st.metric(
                    label="Изменение эффективности",
                    value=diff_text
                )