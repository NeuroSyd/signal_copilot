# This is a feasibility study code pack that aims to produce a Signal To Text for BioSignal Copilot system that we are developing at the 
# School of Biomedical Engineering, The University of Sydney
# Prepared by MEng Student Yongpei Ma, yoma6689@sydney.edu.au in March 2023

import math
import warnings
from statistics import mean

import neurokit2 as nk
import numpy as np
import openai
import wfdb
from keras.models import load_model
from keras.optimizers import Adam
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')

openai.api_key = "<your OpenAI API>"

limb_leads_num = 2
chest_leads_num = 6


def analyze(record):
    ECG_analyze = {}

    signals = record.p_signal.T
    for idx, signal in enumerate(signals):
        ecg_cleaned = nk.ecg_clean(signal, sampling_rate=record.fs)
        # Extract R-peaks locations
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=record.fs)
        signal_rates = nk.signal_rate(rpeaks, sampling_rate=record.fs)
        if np.isnan(signal_rates).all():
            return -1

        ecg_rate = round(mean(signal_rates))
        # print(ecg_rate)
        # # Delineate the ECG signal and visualizing all peaks of ECG complexes
        # _, waves_peak = nk.ecg_delineate(ecg_cleaned, rpeaks, sampling_rate=record.fs, method="peak", show=True, show_type='all')
        # plt.show()

        # Delineate the ECG signal and visualizing all peaks of ECG complexes
        _, waves = nk.ecg_delineate(ecg_cleaned, rpeaks, sampling_rate=record.fs, method="dwt", show=False,
                                    show_type='all')
        # print(waves.keys())
        # print(rpeaks.keys())

        signal_distances = {}
        for index, _ in enumerate(rpeaks['ECG_R_Peaks']):
            P_wave = float(waves['ECG_P_Offsets'][index] - waves['ECG_P_Onsets'][index]) / record.fs
            P_R_segment = float(waves['ECG_R_Onsets'][index] - waves['ECG_P_Offsets'][index]) / record.fs
            P_R_interval = float(waves['ECG_Q_Peaks'][index] - waves['ECG_P_Onsets'][index]) / record.fs
            QRS_complex = float(waves['ECG_S_Peaks'][index] - waves['ECG_Q_Peaks'][index]) / record.fs
            S_T_segment = float(waves['ECG_T_Onsets'][index] - waves['ECG_R_Offsets'][index]) / record.fs
            if math.isnan(waves['ECG_R_Offsets'][index]) or int(
                    waves['ECG_R_Offsets'][index] + record.fs * 0.06) >= len(ecg_cleaned):
                J_60_point = None
            else:
                J_60_point = int(waves['ECG_R_Offsets'][index] + record.fs * 0.06)
            T_wave = float(waves['ECG_T_Offsets'][index] - waves['ECG_T_Onsets'][index]) / record.fs
            Q_T_interval = float(waves['ECG_T_Offsets'][index] - waves['ECG_R_Onsets'][index]) / record.fs
            if index == 0:
                RR_interval = None
            else:
                RR_interval = float(rpeaks['ECG_R_Peaks'][index] - rpeaks['ECG_R_Peaks'][index - 1]) / record.fs

            signal_distances[index] = {'ECG_P_Peaks': waves['ECG_P_Peaks'][index],
                                       'ECG_P_Onsets': waves['ECG_P_Onsets'][index],
                                       'ECG_P_Offsets': waves['ECG_P_Offsets'][index],
                                       'ECG_Q_Peaks': waves['ECG_Q_Peaks'][index],
                                       'ECG_R_Onsets': waves['ECG_R_Onsets'][index],
                                       'ECG_R_Peaks': rpeaks['ECG_R_Peaks'][index],
                                       'ECG_R_Offsets': waves['ECG_R_Offsets'][index],
                                       'ECG_S_Peaks': waves['ECG_S_Peaks'][index],
                                       'ECG_T_Peaks': waves['ECG_T_Peaks'][index],
                                       'ECG_T_Onsets': waves['ECG_T_Onsets'][index],
                                       'ECG_T_Offsets': waves['ECG_T_Offsets'][index], 'P_wave': P_wave,
                                       'P_R_segment': P_R_segment, 'P_R_interval': P_R_interval,
                                       'QRS_complex': QRS_complex, 'S_T_segment': S_T_segment, 'J_60_point': J_60_point,
                                       'T_wave': T_wave, 'Q_T_interval': Q_T_interval, 'RR_interval': RR_interval}

        plt.show()
        # # Zooming into the first 3 R-peaks, with focus on T_peaks, P-peaks, Q-peaks and S-peaks  # nk.events_plot([waves_peak['ECG_T_Peaks'][:3], waves_peak['ECG_P_Peaks'][:3], waves_peak['ECG_Q_Peaks'][:3],  #                 waves_peak['ECG_S_Peaks'][:3]], ecg_cleaned[:record.fs * 3])  # plt.show()

        ECG_analyze[record.sig_name[idx]] = {'ecg_cleaned': ecg_cleaned, 'ecg_rate': ecg_rate,
                                             'signal_distances': signal_distances}

    return ECG_analyze


def getmean_ST(ST_segment_list, ST_segment_level_list):
    ab_ST_segment_list = []
    ab_ST_segment_level_list = []
    n = 0
    i = 0
    while n < 20 and i < len(ST_segment_list):
        if ST_segment_list[i] < 0.08 or ST_segment_list[i] > 0.12:
            ab_ST_segment_list.append(ST_segment_list[i])
            n = n + 1
        i = i + 1

    n = 0
    i = 0
    while n < 20 and i < len(ST_segment_level_list):
        if abs(ST_segment_level_list[i]) > 0.1:
            ab_ST_segment_level_list.append(ST_segment_level_list[i])
            n = n + 1
        i = i + 1

    if len(ab_ST_segment_list) == 0:
        mean_ST_segment = mean(ST_segment_list)
    else:
        mean_ST_segment = mean(ab_ST_segment_list)

    if len(ab_ST_segment_level_list) == 0:
        mean_J_60_point = mean(ST_segment_level_list)
    else:
        mean_J_60_point = mean(ab_ST_segment_level_list)

    return mean_ST_segment, mean_J_60_point


def getmean_all(normal_list, abnormal_list):
    l = (len(normal_list) + len(abnormal_list)) / 2
    if len(abnormal_list) == 0:
        m = np.nansum(normal_list) / len(normal_list)
    else:
        nan_count = np.isnan(abnormal_list).sum()
        zero_count = abnormal_list.count(0)
        if nan_count > l or zero_count > l:
            m = np.nansum(abnormal_list) / (len(normal_list) + len(abnormal_list))
        else:
            m = np.nansum(abnormal_list) / len(abnormal_list)
    return m


def check_diff(lst):
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if abs(lst[i] - lst[j]) >= 0.08:
                return False
    return True


def check_zeros(lst):
    zero_count = lst.count(0)
    if zero_count > (limb_leads_num + chest_leads_num) / 2:
        return ' disappear. '
    else:
        return ' last ' + str(round(sum(lst) / len(lst), 2)) + 's. '


def wave_process(ECG_analyze, lead, t, l):
    RR_interval_list = []

    p_peaks_list = []
    p_duration_list = []
    p_amplitude_list = []
    ab_p_peaks_list = []
    ab_p_duration_list = []
    ab_p_amplitude_list = []

    PR_interval_list = []
    ab_PR_interval_list = []

    QRS_duration_list = []
    ab_QRS_duration_list = []

    R_peaks_list = []
    ab_R_peaks_list = []
    S_peaks_list = []
    ab_S_peaks_list = []
    Q_amplitude_list = []
    ab_Q_amplitude_list = []
    QRS_amplitude_list = []
    ab_QRS_amplitude_list = []

    ST_segment_list = []
    J_60_point_list = []

    T_wave_list = []
    ab_T_wave_list = []
    T_amplitude_list = []
    ab_T_amplitude_list = []

    QT_interval_list = []
    ab_QT_interval_list = []

    ecg_cleaned = ECG_analyze[lead]['ecg_cleaned']
    nan_counter = 0

    for beat_index in ECG_analyze[lead]['signal_distances'].keys():

        # RR interval
        if ECG_analyze[lead]['signal_distances'][beat_index]['RR_interval'] != None:
            RR_interval_list.append(ECG_analyze[lead]['signal_distances'][beat_index]['RR_interval'])

        # P wave
        if ECG_analyze[lead]['signal_distances'][beat_index]['P_wave'] <= 0.12 and \
                ECG_analyze[lead]['signal_distances'][beat_index]['P_wave'] > 0.08:
            p_wave = ECG_analyze[lead]['signal_distances'][beat_index]['P_wave']
            p_duration_list.append(p_wave)
        elif math.isnan(ECG_analyze[lead]['signal_distances'][beat_index]['ECG_P_Peaks']):
            p_wave = 0
            ab_p_duration_list.append(p_wave)
        else:
            ab_p_duration_list.append(ECG_analyze[lead]['signal_distances'][beat_index]['P_wave'])

        if not math.isnan(ECG_analyze[lead]['signal_distances'][beat_index]['ECG_P_Peaks']):
            p_peak = ecg_cleaned[ECG_analyze[lead]['signal_distances'][beat_index]['ECG_P_Peaks']]
            amplitude = abs(ecg_cleaned[ECG_analyze[lead]['signal_distances'][beat_index]['ECG_P_Peaks']]) * 4
            if lead == "DII" and p_peak < 0:
                ab_p_peaks_list.append(p_peak)
            else:
                p_peaks_list.append(p_peak)
        else:
            amplitude = 0

        if amplitude <= 0.5 and amplitude >= 0.1:
            p_amplitude_list.append(amplitude)
        else:
            ab_p_amplitude_list.append(amplitude)

        # PR interval
        if ECG_analyze[lead]['signal_distances'][beat_index]['P_R_interval'] >= 0.12 and \
                ECG_analyze[lead]['signal_distances'][beat_index]['P_R_interval'] <= 0.22:
            PR_interval_list.append(ECG_analyze[lead]['signal_distances'][beat_index]['P_R_interval'])
        else:
            ab_PR_interval_list.append(ECG_analyze[lead]['signal_distances'][beat_index]['P_R_interval'])

        # QRS Complex
        R_peak = ecg_cleaned[ECG_analyze[lead]['signal_distances'][beat_index]['ECG_R_Peaks']] * 2.5
        if not math.isnan(ECG_analyze[lead]['signal_distances'][beat_index]['ECG_S_Peaks']):
            S_peak = ecg_cleaned[ECG_analyze[lead]['signal_distances'][beat_index]['ECG_S_Peaks']] * 2.5

            if R_peak > 0:
                R_peaks_list.append(R_peak)
            else:
                ab_R_peaks_list.append(R_peak)

            if S_peak < 0:
                S_peaks_list.append(S_peak)
            elif S_peak > 0:
                ab_S_peaks_list.append(S_peak)

            amplitude = R_peak - S_peak
            if lead in ['V1', 'V5']:
                if amplitude <= 3.5:
                    QRS_amplitude_list.append(amplitude)
                else:
                    ab_QRS_amplitude_list.append(amplitude)
            elif lead == 'V6':
                if R_peak <= 2.6:
                    QRS_amplitude_list.append(amplitude)
                else:
                    ab_QRS_amplitude_list.append(amplitude)
            else:
                if amplitude <= 4.5:
                    QRS_amplitude_list.append(amplitude)
                else:
                    ab_QRS_amplitude_list.append(amplitude)

            if not math.isnan(ECG_analyze[lead]['signal_distances'][beat_index]['ECG_Q_Peaks']):
                Q_peak = ecg_cleaned[ECG_analyze[lead]['signal_distances'][beat_index]['ECG_Q_Peaks']]
                if Q_peak < 0:
                    Q_amplitude_list.append(Q_peak)
                else:
                    ab_Q_amplitude_list.append(Q_peak)
            else:
                Q_peak = 0
                if lead in ['DI', 'aVL', 'V5', 'V6']:
                    ab_Q_amplitude_list.append(Q_peak)
                else:
                    Q_amplitude_list.append(Q_peak)

            if ECG_analyze[lead]['signal_distances'][beat_index]['QRS_complex'] > 0.06 and \
                    ECG_analyze[lead]['signal_distances'][beat_index]['QRS_complex'] <= 0.12:
                QRS_duration_list.append(ECG_analyze[lead]['signal_distances'][beat_index]['QRS_complex'])
            else:
                ab_QRS_duration_list.append(ECG_analyze[lead]['signal_distances'][beat_index]['QRS_complex'])

        # ST segment
        if not math.isnan(ECG_analyze[lead]['signal_distances'][beat_index]['S_T_segment']):
            ST_segment_list.append(ECG_analyze[lead]['signal_distances'][beat_index]['S_T_segment'])
        else:
            ST_segment_list.append(0)
        if ECG_analyze[lead]['signal_distances'][beat_index]['J_60_point'] == None:
            ST_segment_level = 0
        else:
            ST_segment_level = ecg_cleaned[ECG_analyze[lead]['signal_distances'][beat_index]['J_60_point']]
        J_60_point_list.append(ST_segment_level)

        # T waves

        if not math.isnan(ECG_analyze[lead]['signal_distances'][beat_index]['T_wave']):
            T_wave = ECG_analyze[lead]['signal_distances'][beat_index]['T_wave']
            T_peak = ecg_cleaned[ECG_analyze[lead]['signal_distances'][beat_index]['ECG_T_Peaks']]
        else:
            nan_counter = nan_counter + 1
            T_wave = 0
            T_peak = 0
        if T_wave >= 0.16 and T_wave <= 0.24:
            T_wave_list.append(T_wave)
        else:
            ab_T_wave_list.append(T_wave)

        if lead in ['DI', 'DII', 'V4', 'V5', 'V6'] and T_peak > 0:
            T_amplitude_list.append(T_peak)
        elif lead == 'aVR' and T_peak < 0:
            T_amplitude_list.append(T_peak)
        elif lead in ['DIII', 'aVF', 'aVL', 'V1', 'V2', 'V3']:
            T_amplitude_list.append(T_peak)
        else:
            ab_T_amplitude_list.append(T_peak)

        # QT interval
        QT_interval = ECG_analyze[lead]['signal_distances'][beat_index]['Q_T_interval']
        if math.isnan(QT_interval):
            QT_interval = 0
            ab_QT_interval_list.append(QT_interval)
        elif QT_interval > t:
            ab_QT_interval_list.append(QT_interval)
        else:
            QT_interval_list.append(QT_interval)

    # output

    if l < 20:
        nan_count = np.isnan(ab_p_duration_list).sum()
        if nan_count > l / 2:
            mean_p_wave_duration = np.nan
        else:
            mean_p_wave_duration = (np.nansum(p_duration_list) + np.nansum(ab_p_duration_list)) / l
        if not math.isnan(mean_p_wave_duration):
            mean_p_peak = (np.nansum(p_peaks_list) + np.nansum(ab_p_peaks_list)) / l
        else:
            mean_p_peak = np.nan

        mean_PR_interval = (np.nansum(PR_interval_list) + np.nansum(ab_PR_interval_list)) / l

        mean_QRS_duration = (np.nansum(QRS_duration_list) + np.nansum(ab_QRS_duration_list)) / l
        mean_QRS_amplitude = (np.nansum(QRS_amplitude_list) + np.nansum(ab_QRS_amplitude_list)) / l
        mean_Q_amplitude = (np.nansum(Q_amplitude_list) + np.nansum(ab_Q_amplitude_list)) / l
        mean_R_peaks = (np.nansum(R_peaks_list) + np.nansum(ab_R_peaks_list)) / l
        mean_S_peaks = (np.nansum(S_peaks_list) + np.nansum(ab_S_peaks_list)) / l

        mean_ST_segment = mean(ST_segment_list)
        mean_J_60_point = mean(J_60_point_list)

        if nan_counter > len(ECG_analyze[lead]['signal_distances'][beat_index]) / 2:
            mean_T_waves, mean_T_amplitude, mean_QT_interval = 0, 0, 0
        else:
            mean_T_waves = (np.nansum(T_wave_list) + np.nansum(ab_T_wave_list)) / l
            mean_T_amplitude = (np.nansum(T_amplitude_list) + np.nansum(ab_T_amplitude_list)) / l
            mean_QT_interval = (np.nansum(QT_interval_list) + np.nansum(ab_QT_interval_list)) / l
    else:
        nan_count = np.isnan(ab_p_duration_list).sum()
        if nan_count > l / 2:
            mean_p_wave_duration = np.nan
        else:
            mean_p_wave_duration = getmean_all(p_duration_list, ab_p_duration_list)
        if not math.isnan(mean_p_wave_duration):
            mean_p_peak = getmean_all(p_peaks_list, ab_p_peaks_list)
        else:
            mean_p_peak = None

        mean_PR_interval = getmean_all(PR_interval_list, ab_PR_interval_list)

        mean_QRS_duration = getmean_all(QRS_duration_list, ab_QRS_duration_list)
        mean_QRS_amplitude = getmean_all(QRS_amplitude_list, ab_QRS_amplitude_list)
        mean_Q_amplitude = getmean_all(Q_amplitude_list, ab_Q_amplitude_list)
        mean_R_peaks = getmean_all(R_peaks_list, ab_R_peaks_list)
        mean_S_peaks = getmean_all(S_peaks_list, ab_S_peaks_list)

        if nan_counter > len(ECG_analyze[lead]['signal_distances'][beat_index]) / 2:
            mean_T_waves, mean_T_amplitude, mean_QT_interval = 0, 0, 0
        else:
            mean_T_waves = getmean_all(T_wave_list, ab_T_wave_list)
            mean_T_amplitude = getmean_all(T_amplitude_list, ab_T_amplitude_list)
            mean_QT_interval = getmean_all(QT_interval_list, ab_QT_interval_list)

        mean_ST_segment, mean_J_60_point = getmean_ST(ST_segment_list, J_60_point_list)

    return RR_interval_list, mean_p_peak, mean_p_wave_duration, mean_PR_interval, mean_QRS_duration, mean_QRS_amplitude, mean_Q_amplitude, mean_R_peaks, mean_S_peaks, mean_ST_segment, mean_J_60_point, mean_T_waves, mean_T_amplitude, mean_QT_interval


def convert2text(record, sex, age, ECG_analyze):
    if sex == 'F':
        t = 0.47
        stxt = 'female'
    else:
        t = 0.45
        stxt = 'male'

    l = len(ECG_analyze['DI']['signal_distances'])
    R_list = []

    rhythm = []

    P_waves_list = []
    P_peaks_list = []

    PR_interval_list = []

    QRS_complex_list = []
    limb_amplitude_list = []
    chest_amplitude_list = []
    Q_peaks_list = []
    R_peaks_list = []

    ST_segments_list = []
    J_60_points_list = []

    T_waves_list = []
    T_peaks_1_list = []
    T_peaks_2_list = []
    T_peaks_3_list = []

    QT_interval_list = []

    description = {'Heart_rate': ECG_analyze['DI']['ecg_rate'], 'Rhythm': {}}

    for lead in record.sig_name:
        # print("This is the ECG in {} lead.".format(lead))

        RR_interval_list, mean_p_peak, mean_p_wave_duration, mean_PR_interval, mean_QRS_duration, mean_QRS_amplitude, mean_Q_amplitude, mean_R_peaks, mean_S_peaks, mean_ST_segment, mean_J_60_point, mean_T_waves, mean_T_amplitude, mean_QT_interval = wave_process(
            ECG_analyze, lead, t, l)

        # Rhythm description
        rhythm = rhythm + RR_interval_list

        # P wave and PR interval description
        if math.isnan(mean_p_wave_duration):
            p_wave = 0
            p_peak = 0
            PR_interval = 0
        else:
            p_wave = mean_p_wave_duration
            p_peak = mean_p_peak
            PR_interval = mean_PR_interval
        P_waves_list.append(p_wave)
        P_peaks_list.append(p_peak)
        PR_interval_list.append(PR_interval)

        # QRS Complex description
        QRS_complex_list.append(mean_QRS_duration)

        if lead in ['DI', 'DII', 'DIII', 'aVL', 'aVR', 'aVF']:
            limb_amplitude_list.append(mean_QRS_amplitude)
        else:
            chest_amplitude_list.append(mean_QRS_amplitude)

        Q_peaks_list.append(mean_Q_amplitude)

        if lead == 'V1':
            S_peak_V1 = mean_S_peaks
        elif lead == 'V5':
            R_peak_V5 = mean_R_peaks
        elif lead == 'V6':
            R_peak_V6 = mean_R_peaks
        elif lead in ['DI', 'DII', 'DIII']:
            R_list.append(mean_R_peaks)
        elif lead == 'aVL':
            R_peak_aVL = mean_R_peaks
        else:
            R_peaks_list.append(mean_R_peaks)
        R_I_II_III = max(R_list)

        # ST segment description
        ST_segments_list.append(mean_ST_segment)
        J_60_points_list.append(mean_J_60_point)

        # T waves description
        if mean_T_waves == 0:
            T_wave = 0
            QT_interval = 0
        else:
            T_wave = mean_T_waves
            QT_interval = mean_QT_interval
            if lead in ['DI', 'DII', 'V4', 'V5', 'V6']:
                T_peaks_1_list.append(mean_T_amplitude)
            elif lead == 'aVR':
                T_peaks_2_list.append(mean_T_amplitude)
            else:
                T_peaks_3_list.append(mean_T_amplitude)
        T_waves_list.append(T_wave)
        QT_interval_list.append(QT_interval)

    Text = 'Heart rate is ' + str(description['Heart_rate']) + '. '
    Text = Text + 'R-R intervals are {} s with a variance of {}. '.format(round(mean(rhythm), 2),
                                                                          round(np.var(rhythm), 3))

    if check_zeros(P_waves_list) == ' disappear. ':
        Text = Text + 'P_waves' + check_zeros(P_waves_list)
    else:
        Text = Text + 'P_waves' + check_zeros(P_waves_list).replace('. ', ' with {} mV. '.format(
            round(mean(P_peaks_list), 2)))
        Text = Text + 'PR_intervals' + check_zeros(PR_interval_list)

    Text = Text + 'The QRS complex last {} s with {} mV in limb leads and {} mV in chest leads. ' \
                  'In addition, the amplitude of R waves in leads V5 and V6 are {} and {} mV. The maximum ' \
                  'of R-wave amplitude in leads I, II and III is {} mV. '.format(round(mean(QRS_complex_list), 2),
                                                                                 round(mean(limb_amplitude_list), 2),
                                                                                 round(mean(chest_amplitude_list), 2),
                                                                                 round(R_peak_V5, 2),
                                                                                 round(R_peak_V6, 2),
                                                                                 round(R_I_II_III, 2))
    if limb_leads_num + chest_leads_num == 12:
        Text = Text + 'R-wave amplitude in lead aVL is {}. '.format(round(R_peak_aVL, 2))
    Text = Text + 'The other R-wave amplitude in other leads is {} mV. S-wave amplitude in V1 is {} ' \
                  'mV. Q-wave amplitude is {} mV. '.format(round(mean(R_peaks_list), 2), round(S_peak_V1, 2),
                                                           round(mean(Q_peaks_list), 2))

    Text = Text + 'The ST_segments last {} s and J-60 points are {} mV. '.format(round(mean(ST_segments_list), 2),
                                                                                 round(mean(J_60_points_list), 2))

    if check_zeros(T_waves_list) == ' disappear. ':
        Text = Text + 'T_waves' + check_zeros(T_waves_list)
    else:
        if limb_leads_num + chest_leads_num == 12:
            Text = Text + 'T_waves' + check_zeros(P_waves_list) + 'The T-wave amplitude in aVR ' \
                                                                  'is {} mV and that in other leads are {} ' \
                                                                  'mV. '.format(round(mean(T_peaks_2_list), 2), round(
                (sum(T_peaks_1_list) + sum(T_peaks_3_list)) / 11, 2))
        else:
            Text = Text + 'T_waves' + check_zeros(P_waves_list).replace('. ', ' with {} mV. '.format(
                round((sum(T_peaks_1_list) + sum(T_peaks_3_list)) / 8, 2)))
        Text = Text + 'QT_intervals' + check_zeros(QT_interval_list)

    Text = Text.replace('_', ' ')

    return "This is a {}-lead ECG of {}-year-old {}. ".format(limb_leads_num + chest_leads_num, age, stxt) + Text


def run(data_path, sex, age, record, predicts):
    print(record.sig_name)



    ECG_analyze = analyze(record)
    if ECG_analyze == -1:
        return -1

    user_input = convert2text(record, sex, age, ECG_analyze)
    print(user_input)
    print()
    user_input = user_input + ' Does this person have any disease?'

    messages = [{"role": "user",
                 "content": "You are a Medical AI Assistant. You can find reasonable explanations online "
                            "and in your knowledge base based on user descriptions of ECGs and answer the user whether "
                            "there is any disease. For example, 1st degree AV block (1dAVb), right bundle branch block "
                            "(RBBB), left bundle branch block (LBBB), sinus bradycardia (SB), atrial fibrillation (AF) "
                            "and sinus tachycardia (ST). You must provide an interpretation in the first sentence and then answer the reason."}]



    messages.append({"role": "user", "content": user_input})
    different = True
    while (different):
        chat = openai.ChatCompletion.create(model="gpt-4-0613", messages=messages, temperature=0)
        reply = chat.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        first_sentence = reply.split('.')[0]
        print('first_sentence')
        print(first_sentence)
        for predict in predicts:
            if predict in first_sentence:
                different = False

        if different:
            messages.append({"role": "user",
                             "content": "Your interpretation is wrong. Please provide a new interpretation from 1dAVb, RBBB, LBBB, SB, AF, ST, and normal, and please do not give the same interpretation repeatedly. You must provide the interpretation in the first sentence and then answer the reason."})

    print('assistant reply:\n {}\n'.format(reply))

    while user_input != 'stop':
        user_input = input('Please enter the content you want to reply, input stop will exit the program:\n')
        messages.append({"role": "user", "content": user_input})
        chat = openai.ChatCompletion.create(model="gpt-4-0613", messages=messages, temperature=0)
        reply = chat.choices[0].message.content
        print('assistant reply:\n {}\n'.format(reply))
        messages.append({"role": "assistant", "content": reply})


if __name__ == '__main__':
    predict_map = {'1dAVb,1st degree AV block': 0.05, 'RBBB,right bundle branch block': 0.05,
                   'LBBB,left bundle branch block': 0.05, 'SB,sinus bradycardia': 0.05,
                   'AF,atrial fibrillation': 0.05, 'ST,sinus tachycardia': 0.05}
    model = load_model('model.hdf5', compile=False)
    model.compile(loss='binary_crossentropy', optimizer=Adam())

    # #### Path of a WFDB entry ####
    # data_path = 'TNMG7_N1'  # normal     F      26

    data_path = input('Please enter local data path (for example: TNMG7_N1): ')
    sex = input('Please enter the gender of the user the data belongs to (F: female, M: male): F/M ')
    age = input('Please enter the user\'s age (a number, eg: 26): ')

    record = wfdb.rdrecord(data_path)
    data = np.pad(record.p_signal, ((0, 4096 - record.p_signal.shape[0]), (0, 12 - record.p_signal.shape[1])),
                  'constant', constant_values=(0, 0))
    y_score = model.predict(np.asarray([data]), verbose=1)
    predicts = []
    for index, key in enumerate(predict_map.keys()):
        if y_score[0][index] >= predict_map[key]:
            predicts.extend(key.split(','))
    if len(predicts) == 0:
        predicts.append('normal')

    run(data_path, sex, age, record, predicts)
