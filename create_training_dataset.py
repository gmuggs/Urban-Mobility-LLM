import pandas as pd
import random
from datetime import date
import argparse
from generate_system_prompt import generate_train_context
from datetime import datetime
from sample_population import (
    sample_date,
    sample_from_sf_popn
)
from get_acs_data_files import get_acs_data_files


parser = argparse.ArgumentParser(
    description='Similate a survey using LLMs.'
)
parser.add_argument(
    '--size',
    dest='size',
    default=1000,
    type=str,
)
parser.add_argument(
    '--file_name',
    dest='file_name',
    default='training_dataset.csv',
    type=str,
)
parser.add_argument(
    '--exclude_inference_cities',
    dest='exclude_inference_cities',
    default=False,
    type=bool,
)
args = parser.parse_args()

data_trip = pd.read_csv('dataset/NHTS_2017_csv/trippub.csv')
cities_info = pd.read_csv('dataset/NHTS_2017_csv/cities_info.csv')
data_trip = data_trip.loc[~(data_trip.HH_CBSA == 'XXXXX')]
cbsas = cities_info['CBSA'].to_list()
cities_cbsa_to_remov = [
    41860,  # SF
    47900,  # DC
    19100,  # DFW
    31080,  # La
    33460,  # Minneapolis
]
if args.exclude_inference_cities:
    cbsas = [
        i
        for i in cbsas
        if i not in cities_cbsa_to_remov
    ]

data_trip = data_trip.loc[data_trip.HH_CBSA.astype(int).isin(cbsas)]

datas = []

for i in range(int(args.size)):
    try:
        house_id = random.sample(data_trip.HOUSEID.unique().tolist(), 1)[0]
        person_id = random.sample(
            data_trip.loc[
                data_trip.HOUSEID == house_id
            ].PERSONID.unique().tolist(), 1
        )[0]
        a = data_trip.loc[
            (data_trip.HOUSEID == house_id) & (data_trip.PERSONID == person_id)
        ]
        cbsa = int(a['HH_CBSA'].iloc[0])
        location = cities_info.loc[
            cities_info.CBSA == cbsa
        ].iloc[0]['CBSA_Title']
        B01001_loc, S1401_loc, S2301_loc, S2401_loc, S1201_loc, S1101_loc = \
            get_acs_data_files('San Francisco, CA', year=2017)
        (
            sample_sex,
            sample_age_group,
            sample_age,
            sample_race,
            sample_enrolled_in_school,
            sample_participation_in_labor_force,
            sample_employed,
            sample_occupation,
            sample_marital_status,
            sample_household_type,
            sample_own_child_under_18,
            sample_own_child_under_18_type,
        ) = sample_from_sf_popn(
            B01001_loc,
            S1401_loc,
            S2301_loc,
            S2401_loc,
            S1201_loc,
            S1101_loc,
            location='San Francisco, CA',
            year=2017
        )
        start_date = date(2016, 4, 19)
        end_date = date(2017, 4, 25)
        sample_age = random.randint(18, 70)
        (
            survey_date,
            survey_date_weekday
        ) = sample_date(start_date, end_date)
        context = generate_train_context(
            sample_sex,
            sample_age,
            sample_race,
            location,
            survey_date,
            survey_date_weekday,
            'none'
        )
        # print(context)
        # a
        dnames = {
            1: "Home",
            2: "Work from Home",
            3: "Work",
            4: "Work-related",
            5: "Volunteer",
            6: "Drop-off/Pick-up",
            7: "Change transport",
            8: "School-student",
            9: "Child care",
            10: "Adult care",
            11: "Buy goods",
            12: "Buy services",
            13: "Buy meals",
            14: "General",
            15: "Recreation",
            16: "Exercise",
            17: "Visit friends",
            18: "Health care",
            19: "Religious",
            97: "Other",
        }
        output = ''
        output += (
            "| Place Visited           | Arrival Time    | Departure Time  | "
            "Location Type   |\n|-------------------------|-----------------|-----"  # noqa: E501
            "------------|-----------------|\n"
        )
        place_name_len = len(' [Place Name]            ')
        arrival_time_len = len(' [HH:MM AM/PM]   ')
        departure_time_len = len(' [HH:MM AM/PM]   ')
        location_type_len = len(' [Location Type] ')

        for i, row in enumerate(a.iterrows()):
            # print(i)
            # print(f'INPUT')
            if i == 0:
                # print('->', row[1]['WHYFROM'], '00:00' ,row[1]['STRTTIME'])
                try:
                    place_name = (
                        f" {dnames[row[1]['WHYFROM']]}" +
                        ' ' * (
                            place_name_len - len(dnames[row[1]['WHYFROM']])-1
                        )
                    )
                except:  # noqa: E722
                    place_name = (
                        " Other" +
                        ' ' * (place_name_len - len('Other')-1)
                    )
                t = datetime.strptime('00:00', '%H:%M')
                formatted_time = t.strftime("%I:%M %p")
                arrival_time = (
                    f' {formatted_time}' +
                    ' ' * (arrival_time_len - len(formatted_time)-1)
                )
                t = datetime.strptime(str(row[1]['STRTTIME']), '%H%M')
                formatted_time = t.strftime("%I:%M %p")
                departure_time = (
                    f' {formatted_time}' +
                    ' ' * (departure_time_len - len(formatted_time)-1)
                )
                loc_type = (
                    f" {row[1]['WHYFROM']}" +
                    ' ' * (location_type_len - len(str(row[1]['WHYFROM']))-1)
                )
                output += (
                    f"|{place_name}|{arrival_time}|"
                    f"{departure_time}|{loc_type}|\n"
                )
                # print(
                #     '->',
                #     row[1]['WHYTO'],
                #     row[1]['ENDTIME'],
                #     a.iloc[i+1]['STRTTIME']
                # )
                try:
                    place_name = (
                        f" {dnames[row[1]['WHYTO']]}" +
                        ' ' * (place_name_len - len(dnames[row[1]['WHYTO']])-1)
                    )
                except:  # noqa: E722
                    place_name = (
                        " Other" +
                        ' ' * (place_name_len - len('Other')-1)
                    )
                t = datetime.strptime(str(row[1]['ENDTIME']), '%H%M')
                formatted_time = t.strftime("%I:%M %p")
                arrival_time = (
                    f' {formatted_time}' +
                    ' ' * (arrival_time_len - len(formatted_time)-1)
                )
                if i == len(a)-1:
                    t = datetime.strptime('23:59', '%H:%M')
                else:
                    t = datetime.strptime(str(a.iloc[i+1]['STRTTIME']), '%H%M')
                formatted_time = t.strftime("%I:%M %p")
                departure_time = (
                    f' {formatted_time}' +
                    ' ' * (departure_time_len - len(formatted_time)-1)
                )
                loc_type = (
                    f" {row[1]['WHYTO']}" +
                    ' ' * (location_type_len - len(str(row[1]['WHYTO']))-1)
                )
                output += (
                    f"|{place_name}|{arrival_time}"
                    f"|{departure_time}|{loc_type}|\n"
                )
            elif i == len(a)-1:
                # print('->', row[1]['WHYTO'], row[1]['ENDTIME'], '23:59')
                try:
                    place_name = (
                        f" {dnames[row[1]['WHYTO']]}" +
                        ' ' * (place_name_len - len(dnames[row[1]['WHYTO']])-1)
                    )
                except:  # noqa: E722
                    place_name = (
                        " Other" +
                        ' ' * (place_name_len - len('Other')-1)
                    )
                t = datetime.strptime(str(row[1]['ENDTIME']), '%H%M')
                formatted_time = t.strftime("%I:%M %p")
                arrival_time = (
                    f' {formatted_time}' +
                    ' ' * (arrival_time_len - len(formatted_time)-1)
                )
                t = datetime.strptime('23:59', '%H:%M')
                formatted_time = t.strftime("%I:%M %p")
                departure_time = (
                    f' {formatted_time}' +
                    ' ' * (departure_time_len - len(formatted_time)-1)
                )
                loc_type = (
                    f" {row[1]['WHYTO']}" +
                    ' ' * (location_type_len - len(str(row[1]['WHYTO']))-1)
                )
                output += (
                    f"|{place_name}|{arrival_time}"
                    f"|{departure_time}|{loc_type}|\n"
                )
            else:
                # print(
                #     '->',
                #     row[1]['WHYTO'],
                #     a.iloc[i]['ENDTIME'],
                #     a.iloc[i+1]['STRTTIME']
                # )
                try:
                    place_name = (
                        f" {dnames[row[1]['WHYTO']]}" +
                        ' ' * (place_name_len - len(dnames[row[1]['WHYTO']])-1)
                    )
                except:  # noqa: E722
                    place_name = (
                        " Other" +
                        ' ' * (place_name_len - len('Other')-1)
                    )
                t = datetime.strptime(str(row[1]['ENDTIME']), '%H%M')
                formatted_time = t.strftime("%I:%M %p")
                arrival_time = (
                    f' {formatted_time}' +
                    ' ' * (arrival_time_len - len(formatted_time)-1)
                )
                t = datetime.strptime(str(a.iloc[i+1]['STRTTIME']), '%H%M')
                formatted_time = t.strftime("%I:%M %p")
                departure_time = (
                    f' {formatted_time}' +
                    ' ' * (departure_time_len - len(formatted_time)-1)
                )
                loc_type = (
                    f" {row[1]['WHYTO']}" +
                    ' ' * (location_type_len - len(str(row[1]['WHYTO']))-1)
                )
                output += (
                    f"|{place_name}|{arrival_time}|"
                    f"{departure_time}|{loc_type}|\n"
                )
        datas.append({
            'context': context,
            'output': output
        })
    except:  # noqa: E722
        pass
datas = pd.DataFrame(datas)
datas.to_csv(args.file_name, index=True)
