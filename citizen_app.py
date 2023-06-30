import streamlit as st
from streamlit import session_state as state
import matplotlib.pyplot as plt
from main import main, input_ranges, need
import datetime as dt
import pandas as pd
import time
import numpy as np

base_proposal = {
    'n_cars': 300,
    'weekday_time': (dt.datetime(1990, 1, 1, 8, 0), dt.datetime(1990, 1, 2, 1, 0)),
    'holiday_time': (dt.datetime(1990, 1, 1, 8, 0), dt.datetime(1990, 1, 2, 1, 0)),
    'charge_on_road': True,
    'service_lev': 5,
    'once_fare': 500,
}

if "started" not in state:
    state.started = False

if 'proposing' not in state:
    state.proposing = False

if 'num_of_c' not in state:
    state.num_of_c = 0

if 'num_of_i' not in state:
    state.num_of_i = 0

if 'group' not in state:
    state.group = None

if 'current_proposal' not in state:
    state.current_proposal = [[0, 0], [0, 0]]

if 'saved_proposal' not in state:
    state.saved_proposal = [base_proposal.copy()]*5

if 'base_proposal' not in state:
    state.base_proposal = base_proposal

if 'saving' not in state:
    state.saving = False


def make_it_start():
    with open('results/Id.txt', 'w') as f:
        f.write(f'{state.num_of_c}\n{state.num_of_i}')

    state.started = True
    start_button = st.empty()
    state.starttime = time.time()


def make_it_proposing():
    state.proposing = True


def save_proposal(proposal):
    state.current_proposal = proposal


def make_it_saving():
    state.saving = True


def save_proposal(number, proposal):
    state.saving = True
    state.saved_proposal[number] = proposal


def get_benefit(benefit):
    if state.group == 'A':
        return benefit*0.5, benefit*1.5
    elif state.group == 'B':
        return benefit*0.8, benefit*1.2
    else:
        raise ValueError('グループ番号が設定されていません。')


if 'df' not in state:
    state.df = pd.DataFrame(columns=['提案者', 'time', 'n_cars', 'weekday_starthour', 'weekday_startmin', 'weekday_endhour', 'weekday_endmin',
                            'holiday_starthour', 'holiday_startmin', 'holiday_endhour', 'holiday_endmin', 'charge_on_road', 'service_lev', 'once_fare', 'if_accept'])


# 開始ボタンが押されていない場合、開始ボタンだけを表示
if not state.started:
    state.group = st.radio('グループ番号を選択してください', ('A', 'B'))
    a = st.button('開始', on_click=make_it_start)

else:
    st.sidebar.number = int(st.sidebar.radio(
        '保存した提案', [f'提案{i}' for i in range(1, 6)])[2])-1
    st.sidebar.write('車両数:', state.saved_proposal[st.sidebar.number]['n_cars'])
    st.sidebar.write('平日の運行時間:')
    st.sidebar.write(state.saved_proposal[st.sidebar.number]['weekday_time'][0].strftime(
        "%d日目 %H:%M"), '~', state.saved_proposal[st.sidebar.number]['weekday_time'][1].strftime("%d日目 %H:%M"))
    st.sidebar.write('休日の運行時間:')
    st.sidebar.write(state.saved_proposal[st.sidebar.number]['holiday_time'][0].strftime(
        "%d日目 %H:%M"), '~', state.saved_proposal[st.sidebar.number]['holiday_time'][1].strftime("%d日目 %H:%M"))
    st.sidebar.write(
        '路上充電対応:', '可能' if state.saved_proposal[st.sidebar.number]['charge_on_road'] else '不可能')
    st.sidebar.write(
        'サービス水準:', state.saved_proposal[st.sidebar.number]['service_lev'])
    st.sidebar.write(
        '運賃:', state.saved_proposal[st.sidebar.number]['once_fare'])

    st.title("ワークショップ_住民側")

    n_cars = st.slider('車両数 n_cars', min_value=0, max_value=1000,
                       value=300, step=10)
    n_cars_expander = st.expander('車両数の説明')
    n_cars_expander.write('車両数が増えると：')
    n_cars_expander.write('事業側のコスト:増加（車を多く購入するから）')
    n_cars_expander.write('事業側の売上：増加の可能性がある（より多くの移動需要を満たすことが可能になるから）')
    n_cars_expander.write('住民の利益：増加の可能性がある（より多くの移動需要を満たすことが可能になるから）')

    weekday_time = st.slider('平日の運行時間 weekday time', min_value=dt.datetime(1990, 1, 1, 0, 0), max_value=dt.datetime(
        1990, 1, 2, 23, 59), value=(dt.datetime(1990, 1, 1, 8, 0), dt.datetime(1990, 1, 2, 1, 0)), format="D日目 HH:mm", step=dt.timedelta(minutes=30))

    if weekday_time[1]-weekday_time[0] > dt.timedelta(days=1):
        st.warning('平日の営業時間が24時間を超えてはいけません。')

    holiday_time = st.slider('休日の運行時間 holiday time', min_value=dt.datetime(1990, 1, 1, 0, 0), max_value=dt.datetime(
        1990, 1, 2, 23, 59), value=(dt.datetime(1990, 1, 1, 8, 0), dt.datetime(1990, 1, 2, 1, 0)), format="D日目 HH:mm", step=dt.timedelta(minutes=30))

    if holiday_time[1]-holiday_time[0] > dt.timedelta(days=1):
        st.warning('休日の営業時間が24時間を超えてはいけません。')

    charge_on_road = st.checkbox('路上充電対応可能 charge_on_road', value=True)
    charge_on_road_expander = st.expander('路上充電の説明')
    charge_on_road_expander.write('路上充電が可能になると：')
    charge_on_road_expander.write('事業側のコスト：増加（設備を購入する必要があるから）')
    charge_on_road_expander.write('事業側の売上：増加の可能性がある（より多くの需要を満たすことが可能になるから）')

    charge_on_road_expander.write('住民の利益：増加の可能性がある（より多くの需要を満たすことが可能になるから）')

    service_lev = st.slider('サービス水準 service_lev',
                            min_value=1, max_value=11, value=5)
    service_level_expander = st.expander('サービスレベルの説明')
    service_level_expander.write('サービス水準が高いと：')
    service_level_expander.write('事業側のコスト：増加（人件費がかかるから）')
    service_level_expander.write('サービスへの需要：増加（顧客体験がいいから）')

    once_fare = st.slider('3kmあたりの運賃 once_fare',
                          min_value=0, max_value=3000, value=500)
    once_fare_expander = st.expander('運賃の説明')
    once_fare_expander.write('運賃が高いと：')
    once_fare_expander.write('一回あたりの運賃収入は増えるが、サービスへの需要が減る')

    st.write('現在の合意の設計案:')
    st.write(
        f'住民側の便益：{state.current_proposal[1][0]} ~ {state.current_proposal[1][1]} 万円')

    input_values = {
        'n_cars': n_cars,
        'weekday_starthour': weekday_time[0].hour,
        'weekday_startmin': weekday_time[0].minute,
        'weekday_endhour': weekday_time[1].hour,
        'weekday_endmin': weekday_time[1].minute,
        'holiday_starthour': holiday_time[0].hour,
        'holiday_startmin': holiday_time[0].minute,
        'holiday_endhour': holiday_time[1].hour,
        'holiday_endmin':  holiday_time[1].minute,
        'charge_on_road': charge_on_road,
        'service_lev': service_lev,
        'once_fare': once_fare
    }
    temp_proposal = {
        'n_cars': n_cars,
        'weekday_time': weekday_time,
        'holiday_time': holiday_time,
        'charge_on_road': charge_on_road,
        'service_lev': service_lev,
        'once_fare': once_fare,
    }

    checking = st.button('結果確認', on_click=make_it_proposing)

    if checking or state.proposing or state.saving:

        profit, benefit = main(input_values, input_ranges, need, state.group)
        down, up = get_benefit(benefit)
        st.write(f'住民側の便益：{down:.2f} ~ {up:.2f} 万円')
        state.proposing = False

        st.button('合意した設計案を保存する', on_click=save_proposal, args=(
            [[profit*0.7, profit*1.2], [down, up]],))

        if st.button('提案を左側に保存', on_click=make_it_saving) or state.saving:
            number = st.radio(
                'どれに保存しますか？', [f'提案{i}に保存' for i in range(1, 6)], on_change=make_it_saving)
            number = int(number[2])-1
            state.saving = False
            if st.button('保存', on_click=save_proposal, args=(number, temp_proposal)):
                st.success('保存完了')
                time.sleep(0.8)
                st.experimental_rerun()

    st.button('終了')
