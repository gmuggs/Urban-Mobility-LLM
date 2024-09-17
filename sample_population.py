import random
import pandas as pd
from datetime import timedelta
from generate_system_prompt import (
    generate_completion_prompt,
    generate_completion_prompt_simplified
)
from datetime import date
from get_acs_data_files import get_acs_data_files


def sample_date(start_date, end_date):
    dates = []
    delta = timedelta(days=1)
    while end_date >= start_date:
        dates.append(start_date)
        start_date += delta
    random_date = random.choice(dates)
    if random_date.weekday() == 0:
        return random_date, 'Monday'
    elif random_date.weekday() == 1:
        return random_date, 'Tuesday'
    elif random_date.weekday() == 2:
        return random_date, 'Wednesday'
    elif random_date.weekday() == 3:
        return random_date, 'Thursday'
    elif random_date.weekday() == 4:
        return random_date, 'Friday'
    elif random_date.weekday() == 5:
        return random_date, 'Saturday'
    elif random_date.weekday() == 6:
        return random_date, 'Sunday'
    else:
        return random_date, 'Monday'


def convert_str_to_float(string):
    if type(string) is float:
        return string
    elif type(string) is int:
        return float(string)
    else:
        return float(string.replace(',', ''))


def get_age_from_age_group(age_group):
    if age_group == 'Under 5 years':
        return random.sample([1, 2, 3, 4], k=1)[0]
    elif age_group == '85 years and over':
        return random.sample(range(85, 101), k=1)[0]
    elif age_group == '20 years':
        return 20
    elif age_group == '21 years':
        return 21
    else:
        s = int(age_group.split(' ')[0])
        e = int(age_group.split(' ')[2])
        return random.sample(range(s, e+1), k=1)[0]


def get_age_from_age_group_uk(age_group):
    if age_group == 'Aged 2 years and under':
        return random.sample([1, 2], k=1)[0]
    elif age_group == 'Aged 15 years':
        return 15
    elif age_group == 'Aged 65 years':
        return 65
    elif age_group == 'Aged 85 years and over':
        return random.sample(range(85, 100), k=1)[0]
    else:
        s = int(age_group.split(' ')[1])
        e = int(age_group.split(' ')[3])
        return random.sample(range(s, e+1), k=1)[0]


def sample_from_sf_popn(
    B01001_loc,
    S1401_loc,
    S2301_loc,
    S2401_loc,
    S1201_loc,
    S1101_loc,
    location='San Francisco, CA',
    year=2021
):
    '''
        Takes in the location of the census data files and samples from the
        population of San Francisco.
        Returns:
            sample_sex,
            sample_age_group,
            sample_age,
            sample_race,
            sample_enrolled_in_school,
            sample_participation_in_labor_force,
            sample_employed,
            sample_occupation,
            sample_marital_status,
            sample_own_child_under_18,
            sample_own_child_under_18_type,

    '''
    # SEX BY AGE Dataset
    B01001 = pd.read_csv(B01001_loc)
    l1 = '{}!!Total population!!Estimate'
    l2 = '{}!!Percent!!Estimate'
    l3 = (
        '{}!!'
        'Labor Force Participation Rate!!Estimate'
    )
    l4 = '{}!!Unemployment rate!!Estimate'
    l5 = (
        '{}!!'
        'Married-couple family household!!Estimate'
    )
    l6 = (
        '{}!!'
        'Male householder, no spouse present, family household!!Estimate'
    )
    l6_2 = (
        '{}!!'
        'Male householder, no wife present, family household!!Estimate'
    )
    l7 = (
        '{}!!'
        'Female householder, no spouse present, family household!!Estimate'
    )
    l7_2 = (
        '{}!!'
        'Female householder, no husband present, family household!!Estimate'
    )
    l8 = '{}!!Total!!Estimate'
    tot_pop_text = (
        '{}!!'
        'Total population!!Estimate'
    )

    if year == 2021:
        location_mapper = {
            'San Francisco, CA': 'San Francisco County, California',
            'Chicago, IL': 'Chicago city, Cook County, Illinois',
            'Houston, TX': 'Houston city, Texas',
            'Pittsburgh, PA': 'Pittsburgh city, Pennsylvania',
            'Oklahoma City, OK': 'Oklahoma City, OK Metro Area',
            'Baltimore, MD': 'Baltimore-Columbia-Towson, MD Metro Area'
        }
    elif year == 2017:
        location_mapper = {
            'San Francisco, CA': 'San Francisco-Oakland-Hayward, CA Metro Area',  # noqa: E501
            'Chicago, IL': 'Chicago city, Cook County, Illinois',
            'Houston, TX': 'Houston city, Texas',
            'Pittsburgh, PA': 'Pittsburgh city, Pennsylvania',
            'Oklahoma City, OK': 'Oklahoma City, OK Metro Area',
            'Baltimore, MD': 'Baltimore-Columbia-Towson, MD Metro Area',
            'dc': 'Washington-Arlington-Alexandria, DC-VA-MD-WV Metro Area',
            'dfw': 'Dallas-Fort Worth-Arlington, TX Metro Area',
            'minneapolis': 'Minneapolis-St. Paul-Bloomington, MN-WI Metro Area',  # noqa: E501
            'la': 'Los Angeles-Long Beach-Anaheim, CA Metro Area',
        }
    l1 = l1.format(location_mapper[location])
    l2 = l2.format(location_mapper[location])
    l3 = l3.format(location_mapper[location])
    l4 = l4.format(location_mapper[location])
    l5 = l5.format(location_mapper[location])
    l6 = l6.format(location_mapper[location])
    l6_2 = l6_2.format(location_mapper[location])
    l7 = l7.format(location_mapper[location])
    l7_2 = l7_2.format(location_mapper[location])
    l8 = l8.format(location_mapper[location])
    tot_pop_text = tot_pop_text.format(location_mapper[location])

    male_popn = convert_str_to_float(
        B01001.loc[
            B01001['Label (Grouping)'] == '\xa0\xa0\xa0\xa0Male:'
        ][
            l1
        ].item()
    )
    female_popn = convert_str_to_float(
        B01001.loc[
            B01001['Label (Grouping)'] == '\xa0\xa0\xa0\xa0Female:'
        ][
            l1
        ].item()
    )

    sample_sex = random.choices(
        ['male', 'female'],
        weights=(male_popn, female_popn)
    )[0]
    if sample_sex == 'male':
        start_index = 2
        end_index = 25
    else:
        start_index = 26
        end_index = 59

    df_sex_wise = B01001.iloc[start_index:end_index].copy()
    df_sex_wise[tot_pop_text] = df_sex_wise[tot_pop_text].apply(
        convert_str_to_float
    )
    a = df_sex_wise.sample(weights=df_sex_wise[tot_pop_text])
    sample_age_group = a['Label (Grouping)'].item().lstrip()
    sample_age = get_age_from_age_group(sample_age_group)
    b = a.loc[:, ~a.columns.isin(['Label (Grouping)', tot_pop_text])].copy()
    for column in b.columns:
        b[column] = b[column].apply(convert_str_to_float)
    sample_race = b.sample(
        weights=b.values.tolist()[0], axis=1
    ).columns[0].split("!!")[1]

    # School Enrollment
    S1401 = pd.read_csv(S1401_loc)
    sample_enrolled_in_school = False
    if sample_age <= 2:
        pass
    else:
        c = S1401.iloc[12:28][1::2]
        for k, v in c.iterrows():
            c_a = v['Label (Grouping)'].strip()
            if c_a == '35 years and over enrolled in school':
                s = 35
                e = 100
            else:
                s = int(c_a.split()[0])
                e = int(c_a.split()[2])
            if sample_age in range(s, e+1):
                yes_prob = float(
                    v[
                        l2
                    ].rstrip("%")
                )
                no_prob = 100 - yes_prob
                sample_enrolled_in_school = random.choices(
                    [True, False],
                    weights=(yes_prob, no_prob)
                )[0]

    # Employment Status
    S2301 = pd.read_csv(S2301_loc)

    sample_participation_in_labor_force = False
    sample_employed = False
    if sample_age < 16:
        pass
    else:
        for k, v in S2301.iloc[2:12].iterrows():
            c_a = v['Label (Grouping)'].strip()
            if c_a == '75 years and over':
                s = 75
                e = 100
            else:
                s = int(c_a.split()[0])
                e = int(c_a.split()[2])
            if sample_age in range(s, e+1):
                if sample_sex == 'Male':
                    i = 24
                else:
                    i = 25
                yes_prob = float(
                    v[
                        l3
                    ].rstrip("%")
                ) * float(
                    S2301.iloc[i][
                        l3
                    ].rstrip("%")
                )
                yes_prob /= 100
                no_prob = 100 - yes_prob
                sample_participation_in_labor_force = random.choices(
                    [True, False],
                    weights=(yes_prob, no_prob)
                )[0]
                if sample_participation_in_labor_force:
                    no_prob = float(
                        v[
                            l4
                        ].rstrip("%")
                    ) * float(
                        S2301.iloc[i][
                            l4
                        ].rstrip("%")
                    )
                    yes_prob = 100 - no_prob
                    sample_employed = random.choices(
                        [True, False],
                        weights=(yes_prob, no_prob)
                    )[0]

    # Occupation
    sample_occupation = None
    if sample_employed:
        S2401 = pd.read_csv(S2401_loc)
        lis = [
            [3, 5],
            [6, 9],
            [10, 14],
            [15, 17],
            [18, 19],
            [20, 22],
            [22, 25],
            [26, 28],
            [29, 32],
            [33, 36],
        ]
        indexes = []
        for each in lis:
            for i in range(each[0], each[1]):
                indexes.append(i)
        final_occupations = S2401.iloc[indexes].copy()
        final_occupations['Label (Grouping)'] = final_occupations[
            'Label (Grouping)'
        ].apply(lambda x: x.strip())
        c_a = l8
        final_occupations[c_a] = final_occupations[c_a].apply(
            convert_str_to_float
        )
        final_occupations
        sample_occupation = final_occupations.sample(
            weights=final_occupations[c_a]
        )['Label (Grouping)'].item()

    # Marital Status
    S1201 = pd.read_csv(S1201_loc)
    sample_marital_status = None

    if sample_sex == 'male':
        s = 3
        e = 9
    else:
        s = 10
        e = 16

    if sample_age < 15:
        sample_marital_status = 'never married'
    else:
        marital_status_df = S1201.iloc[s:e].copy()
        marital_status_df['Label (Grouping)'] = marital_status_df[
            'Label (Grouping)'
        ].apply(lambda x: x.strip())
        for k, v in marital_status_df.iterrows():
            c_a = v['Label (Grouping)']
            if c_a == '65 years and over':
                s = 75
                e = 100
            else:
                s = int(c_a.split()[0])
                e = int(c_a.split()[2])
            if sample_age in range(s, e+1):
                sample_marital_status = random.choices(
                    [
                        'married',
                        'widowed',
                        'divorced',
                        'seprated',
                        'never married',
                    ],
                    weights=[
                        float(x.rstrip('%')) for x in v.values.tolist()[2:]
                    ]
                )[0]

    # Family Structure
    S1101 = pd.read_csv(S1101_loc)
    if sample_marital_status == 'married':
        sample_household_type = 'married couple family'
    elif sample_marital_status in ['widowed', 'divorced', 'seprated']:
        sample_household_type = random.choices(
            [
                'male householder, no spouse present, family',
                'female householder, no spouse present, family',
                'non family household',
            ],
            weights=[
                float(
                    convert_str_to_float(x)
                ) for x in S1101.iloc[1].values.tolist()[3:]
            ]
        )[0]
    else:
        sample_household_type = random.choices(
            [
                'married couple family',
                'male householder, no spouse present, family',
                'female householder, no spouse present, family',
                'non family household',
            ],
            weights=[
                float(
                    convert_str_to_float(x)
                ) for x in S1101.iloc[1].values.tolist()[2:]
            ]
        )[0]

    sample_own_child_under_18 = None
    sample_own_child_under_18_type = None
    c_a = None
    if sample_household_type == 'married couple family':
        c_a = l5
    elif sample_household_type in [
        'male householder, no spouse present, family',
        'male householder, no wife present, family',
    ]:
        if year == 2021:
            c_a = l6
        elif year == 2017:
            c_a = l6_2
    elif sample_household_type in [
        'female householder, no spouse present, family'
        'female householder, no husband present, family'
    ]:
        if year == 2021:
            c_a = l7
        elif year == 2017:
            c_a = l7_2
    else:
        c_a = None

    if c_a is not None:
        c_b = S1101[c_a].to_list()
        sample_own_child_under_18 = random.choices(
            [True, False],
            weights=[float(convert_str_to_float(x)) for x in [c_b[7], c_b[1]]]
        )[0]
        if sample_own_child_under_18:
            sample_own_child_under_18_type = random.choices(
                [
                    'under 6 years only',
                    'under 6 years and 6 to 17 years',
                    '6 to 17 years only',
                ],
                weights=[float(x.rstrip('%')) for x in c_b[8:11]]
            )[0]
    return (
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
    )


def sample_from_uk_population(
    TS008_loc,
    RM121_loc,
    TS021_loc,
    TS066_loc,
    RM074_loc
):
    TS008 = pd.read_csv(TS008_loc)
    RM121 = pd.read_csv(RM121_loc)
    TS021 = pd.read_csv(TS021_loc)
    TS066 = pd.read_csv(TS066_loc)
    RM074 = pd.read_csv(RM074_loc)

    male_popn = TS008.loc[
        TS008['Sex (2 categories)'] == 'Male'
    ]['Observation'].item()
    female_popn = TS008.loc[
        TS008['Sex (2 categories)'] == 'Female'
    ]['Observation'].item()

    sample_sex = random.choices(
        ['male', 'female'],
        weights=(male_popn, female_popn)
    )[0]
    if sample_sex == 'male':
        start_index = 23
        end_index = 46
    else:
        start_index = 0
        end_index = 23

    df_sex_wise = RM121.iloc[start_index:end_index]
    sample_age_group = df_sex_wise.sample(
        weights=df_sex_wise['Observation']
    )['Age (23 categories)'].item()
    sample_age = get_age_from_age_group_uk(sample_age_group)
    sample_race = TS021.sample(
        weights=TS021['Observation']
    )['Ethnic group (20 categories)'].item()

    sample_enrolled_in_school = False
    sample_participation_in_labor_force = False
    sample_employed = False
    if sample_age < 17:
        pass
    else:
        TS066 = TS066.loc[
            ~(TS066['Economic activity status (20 categories) Code'] == -8)
        ]
        a = TS066.sample(
            weights=TS066['Observation']
        )['Economic activity status (20 categories) Code'].item()
        if a in list(range(1, 7)):
            sample_enrolled_in_school = False
            sample_participation_in_labor_force = True
            sample_employed = True
        elif a == 7:
            sample_enrolled_in_school = False
            sample_participation_in_labor_force = True
            sample_employed = False
        elif a in list(range(8, 14)):
            sample_enrolled_in_school = True
            sample_participation_in_labor_force = True
            sample_employed = True
        elif a == 14:
            sample_enrolled_in_school = True
            sample_participation_in_labor_force = True
            sample_employed = False
        elif a == 15:
            sample_enrolled_in_school = False
            sample_participation_in_labor_force = False
            sample_employed = False
        elif a == 16:
            sample_enrolled_in_school = True
            sample_participation_in_labor_force = False
            sample_employed = False
        elif a in list(range(17, 20)):
            sample_enrolled_in_school = False
            sample_participation_in_labor_force = False
            sample_employed = False

    if sample_age <= 15:
        age_cat = 'Aged 15 years and under'
    elif sample_age in range(16, 25):
        age_cat = 'Aged 16 to 24 years'
    elif sample_age in range(25, 35):
        age_cat = 'Aged 25 to 34 years'
    elif sample_age in range(35, 50):
        age_cat = 'Aged 35 to 49 years'
    elif sample_age in range(50, 65):
        age_cat = 'Aged 50 to 64 years'
    else:
        age_cat = 'Aged 65 years and over'

    if sample_sex == 'female':
        a = RM074.loc[
            (RM074['Sex (2 categories) Code'] == 1) &
            (RM074['Age (6 categories)'] == age_cat)
        ].copy()
        sample_marital_status = a.sample(
            weights=a['Observation']
        )['Marital and civil partnership status (6 categories)'].item()
    else:
        a = RM074.loc[
            (RM074['Sex (2 categories) Code'] == 2) &
            (RM074['Age (6 categories)'] == age_cat)
        ].copy()
        sample_marital_status = a.sample(
            weights=a['Observation']
        )['Marital and civil partnership status (6 categories)'].item()

    if sample_marital_status == 'Does not apply':
        sample_marital_status = 'Never married'

    return (
        sample_sex,
        sample_age_group,
        sample_age,
        sample_race,
        sample_enrolled_in_school,
        sample_participation_in_labor_force,
        sample_employed,
        sample_marital_status
    )


if __name__ == '__main__':
    location = 'la'
    if location == "Greater London Area (UK)":
        base_loc = (
            'dataset/census_data/greater_london_region/'
        )
        TS008_loc = (
            base_loc +
            'TS008-2021-4-filtered-2023-10-25T14_18_04Z.csv'
        )
        RM121_loc = (
            base_loc +
            'RM121-2021-1-filtered-2023-10-25T14_46_37Z.csv'
        )
        TS021_loc = (
            base_loc +
            'TS021-2021-3-filtered-2023-10-25T14_41_43Z.csv'
        )
        TS066_loc = (
            base_loc +
            'TS066-2021-5-filtered-2023-10-25T21_03_37Z.csv'
        )
        RM074_loc = (
            base_loc +
            'RM074-2021-1-filtered-2023-10-25T21_27_09Z.csv'
        )
    else:
        B01001_loc, S1401_loc, S2301_loc, S2401_loc, S1201_loc, S1101_loc = \
            get_acs_data_files(location, year=2017)

    for i in range(100):
        if location == "Greater London Area (UK)":
            (
                sample_sex,
                sample_age_group,
                sample_age,
                sample_race,
                sample_enrolled_in_school,
                sample_participation_in_labor_force,
                sample_employed,
                sample_marital_status,
            ) = sample_from_uk_population(
                TS008_loc,
                RM121_loc,
                TS021_loc,
                TS066_loc,
                RM074_loc
            )
            print(f'sample_sex: {sample_sex}')
            print(f'sample_age_group: {sample_age_group}')
            print(f'sample_age: {sample_age}')
            print(f'sample_race: {sample_race}')
            print(f'sample_enrolled_in_school: {sample_enrolled_in_school}')
            print(
                f'sample_participation_in_labor_force: '
                f'{sample_participation_in_labor_force}'
            )
            print(f'sample_employed: {sample_employed}')
            print(f'sample_marital_status: {sample_marital_status}')
            start_date = date(2016, 4, 19)
            end_date = date(2017, 4, 25)
            (
                survey_date,
                survey_date_weekday
            ) = sample_date(start_date, end_date)

            print(generate_completion_prompt_simplified(
                sample_sex,
                sample_age,
                sample_race,
                sample_enrolled_in_school,
                sample_participation_in_labor_force,
                sample_employed,
                sample_marital_status,
                location,
                survey_date,
                survey_date_weekday,
            ))
        else:
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
                location=location,
                year=2017
            )
            print(f'sample_sex: {sample_sex}')
            print(f'sample_age_group: {sample_age_group}')
            print(f'sample_age: {sample_age}')
            print(f'sample_race: {sample_race}')
            print(f'sample_enrolled_in_school: {sample_enrolled_in_school}')
            print(
                f'sample_participation_in_labor_force: '
                f'{sample_participation_in_labor_force}'
            )
            print(f'sample_employed: {sample_employed}')
            print(f'sample_occupation: {sample_occupation}')
            print(f'sample_marital_status: {sample_marital_status}')
            print(f'sample_marital_status: {sample_household_type}')
            print(f'sample_own_child_under_18: {sample_own_child_under_18}')
            print(
                'sample_own_child_under_18_type: '
                f'{sample_own_child_under_18_type}'
            )
            start_date = date(2016, 4, 19)
            end_date = date(2017, 4, 25)
            (
                survey_date,
                survey_date_weekday
            ) = sample_date(start_date, end_date)

            print(generate_completion_prompt(
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
                location,
                survey_date,
                survey_date_weekday,
                'none'
            ))
