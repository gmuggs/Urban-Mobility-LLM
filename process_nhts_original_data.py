import pandas as pd
import argparse


def get_processed_original_data(cbsa_code):
    data_trip = pd.read_csv('dataset/NHTS_2017_csv/trippub.csv')
    if cbsa_code == 'ALL':
        df = data_trip
    else:
        df = data_trip.loc[data_trip.HH_CBSA == cbsa_code]

    if len(df) == 0:
        raise ValueError('No data found for the given CBSA code')

    df['FROMTO'] = df[['WHYFROM', 'WHYTO']].values.tolist().copy()

    data_t1 = df.groupby(
        ['HOUSEID', 'PERSONID']
    )['FROMTO'].apply(list).reset_index()

    def get_loc_type_(v):
        final_val = []
        for i, each in enumerate(v['FROMTO']):
            if i == 0:
                final_val.append(each[0])
            else:
                if v['FROMTO'][i-1][1] == each[0]:
                    final_val.append(each[0])
                    if i == len(v['FROMTO']) - 1:
                        final_val.append(each[1])
                else:
                    raise ValueError('Error')
        return final_val

    data_t1['loc_type'] = data_t1.apply(lambda x: get_loc_type_(x), axis=1)

    name_mapper = {
        'HOUSEID': 'hhid',
        'PERSONID': 'person',
        'TRVLCMIN': 'travel_time'
    }
    data_t1.rename(columns=name_mapper, inplace=True)
    data_t1.drop(columns=['FROMTO'], inplace=True)

    data_t2 = df.groupby(
        ['HOUSEID', 'PERSONID']
    )[['TRVLCMIN']].sum().reset_index()
    data_t2.rename(columns=name_mapper, inplace=True)
    df = data_t1.merge(data_t2, on=['hhid', 'person'], how='inner')
    df['travel_time'] = df['travel_time'] * 60
    df['location'] = df.apply(lambda x: len(x['loc_type']), axis=1)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Similate a survey using LLMs.'
    )
    parser.add_argument(
        '--cbsa',
        dest='cbsa',
        type=str,
    )
    parser.add_argument(
        '--file_name',
        dest='file_name',
        type=str,
    )
    args = parser.parse_args()
    df = get_processed_original_data(args.cbsa)
    df.to_pickle(
        f'dataset/NHTS_2017_csv/processed_data/{args.file_name}'
    )
