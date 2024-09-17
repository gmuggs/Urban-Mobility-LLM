
import json
import os
import pandas as pd
import re
import argparse


def process_outputs_new(
    folder_loc='outputs/',
    fix_missing=False
):
    files = os.listdir(folder_loc)
    data = []
    counter = 0
    incomplete_generations = 0
    table_not_found = 0
    for f in files:
        f_name = f.split('.')[0]
        t_data = []
        incomplete_generation = False
        # try:
        if f.split('.')[-1] != 'json':
            continue
        answers = json.load(open(folder_loc+f))
        info = answers[-1]
        ans_output = answers[-2]['ans_output']
        ds = ans_output.split('\n')
        ds = [d.strip() for d in ds]
        a = None
        for i, each in enumerate(ds):
            pattern = re.compile(
                r"\|\s*-+\s*\|\s*-+\s*\|\s*-+\s*\|\s*-+\s*\|",
                re.IGNORECASE
            )
            m = pattern.match(each)
            if m is None:
                pass
            else:
                a = i
                break
        if a is None:
            table_not_found += 1
            continue

        while True:
            if a+1 >= len(ds):
                break
            elif ds[a+1] == '':
                break
            else:
                t = ds[a+1].split('|')
                if len(t) == 6:
                    temp_data = info.copy()
                    temp_data['id'] = counter
                    temp_data['uuid'] = f_name
                    temp_data['place_name'] = t[1].strip()
                    temp_data['arrival_time'] = t[2].strip()
                    temp_data['departure_time'] = t[3].strip()
                    temp_data['loc_type'] = t[4].strip()
                    t_data.append(temp_data)
                    a += 1
                else:
                    incomplete_generation = True
                    if fix_missing:
                        # print(t[1].strip())
                        temp_data = info.copy()
                        temp_data['id'] = counter
                        temp_data['uuid'] = f_name
                        temp_data['place_name'] = 'Home'
                        temp_data['arrival_time'] = \
                            t_data[-1]['departure_time']
                        temp_data['departure_time'] = '11:59 PM'
                        temp_data['loc_type'] = '1'
                        t_data.append(temp_data)
                        a += 1
                    else:
                        break
        if incomplete_generation:
            incomplete_generations += 1
            # print(f)
            if fix_missing:
                data.extend(t_data)
                counter += 1
            else:
                continue
        else:
            counter += 1
            data.extend(t_data)
        # except Exception as e:
        #     e
    print(f'Complete generations: {counter}')
    print(f'Incomplete generations: {incomplete_generations}')
    print(f'Table not found: {table_not_found}')
    return data


def process_outputs(
        folder_loc='outputs/'
):
    files = os.listdir(folder_loc)
    data = []
    counter = 0
    for f in files:
        try:
            if f.split('.')[-1] != 'json':
                continue
            answers = json.load(open(folder_loc+f))
            info = answers[-1]
            ans_output = answers[-2]['ans_output']
            ds = ans_output.split('\n')
            ds = [d.strip() for d in ds]

            if '| --- | --- | --- |' in ds:
                a = ds.index('| --- | --- | --- |')
            elif '| --- | --- | --- | --- |' in ds:
                a = ds.index('| --- | --- | --- | --- |')
            else:
                raise Exception('No table found')
            while True:
                if ds[a+1] == '':
                    break
                else:
                    t = ds[a+1].split('|')
                    temp_data = info.copy()
                    temp_data['id'] = counter
                    temp_data['place_name'] = t[1].strip()
                    temp_data['arrival_time'] = t[2].strip()
                    temp_data['departure_time'] = t[3].strip()
                    data.append(temp_data)
                    a += 1
            counter += 1
            print(f)
        except Exception as e:
            e
    return data


def process_outputs_completion(
    folder_loc='outputs/',
    fix_missing=False
):
    files = os.listdir(folder_loc)
    data = []
    counter = 0
    incomplete_generations = 0
    table_not_found = 0
    fix_missing = False
    for f in files:
        f_name = f.split('.')[0]
        t_data = []
        incomplete_generation = False
        if f.split('.')[-1] != 'json':
            continue
        answers = json.load(open(folder_loc+f))
        info = answers.copy()

        for a in ['full_output', 'ans_output', 'input']:
            del info[a]

        ans_output = answers['full_output'].replace(answers['input'], '')
        ds = ans_output.split('\n')
        ds = [d.strip() for d in ds]
        a = None
        for i, each in enumerate(ds):
            pattern = re.compile(
                r"\|\s*-+\s*\|\s*-+\s*\|\s*-+\s*\|\s*-+\s*\|",
                re.IGNORECASE
            )
            m = pattern.match(each)
            if m is None:
                pass
            else:
                a = i
                break
        if a is None:
            table_not_found += 1
            continue

        while True:
            if a+1 >= len(ds):
                break
            elif ds[a+1] == '':
                break
            else:
                t = ds[a+1].split('|')
                if len(t) == 6:
                    temp_data = info.copy()
                    temp_data['id'] = counter
                    temp_data['uuid'] = f_name
                    temp_data['place_name'] = t[1].strip()
                    temp_data['arrival_time'] = t[2].strip()
                    temp_data['departure_time'] = t[3].strip()
                    temp_data['loc_type'] = t[4].strip()
                    t_data.append(temp_data)
                    a += 1
                else:
                    incomplete_generation = True
                    if fix_missing is True:
                        print('Hi')
                        # print(t[1].strip())
                        temp_data = info.copy()
                        temp_data['id'] = counter
                        temp_data['uuid'] = f_name
                        temp_data['place_name'] = 'Home'
                        temp_data['arrival_time'] = \
                            t_data[-1]['departure_time']
                        temp_data['departure_time'] = '11:59 PM'
                        temp_data['loc_type'] = '1'
                        t_data.append(temp_data)
                        a += 1
                    else:
                        break
        if incomplete_generation:
            incomplete_generations += 1
            # print(f)
            if fix_missing is True:
                data.extend(t_data)
                counter += 1
            else:
                continue
        else:
            counter += 1
            data.extend(t_data)
        # except Exception as e:
        #     e
    print(f'Complete generations: {counter}')
    print(f'Incomplete generations: {incomplete_generations}')
    print(f'Table not found: {table_not_found}')
    return data


def process_outputs_completion_palm(
    folder_loc='outputs/',
    fix_missing=False
):
    files = os.listdir(folder_loc)
    data = []
    counter = 0
    incomplete_generations = 0
    table_not_found = 0
    fix_missing = False
    for f in files:
        f_name = f.split('.')[0]
        t_data = []
        incomplete_generation = False
        if f.split('.')[-1] != 'json':
            continue
        answers = json.load(open(folder_loc+f))
        info = answers.copy()

        for a in ['ans_output', 'input']:
            del info[a]

        ans_output = answers['ans_output']
        ds = ans_output.split('\n')
        ds = [d.strip() for d in ds]
        a = None
        for i, each in enumerate(ds):
            pattern = re.compile(
                r"\|\s*-+\s*\|\s*-+\s*\|\s*-+\s*\|\s*-+\s*\|",
                re.IGNORECASE
            )
            m = pattern.match(each)
            if m is None:
                pass
            else:
                a = i
                break
        if a is None:
            table_not_found += 1
            continue

        while True:
            if a+1 >= len(ds):
                break
            elif ds[a+1] == '':
                break
            else:
                t = ds[a+1].split('|')
                if len(t) == 6:
                    temp_data = info.copy()
                    temp_data['id'] = counter
                    temp_data['uuid'] = f_name
                    temp_data['place_name'] = t[1].strip()
                    temp_data['arrival_time'] = t[2].strip()
                    temp_data['departure_time'] = t[3].strip()
                    temp_data['loc_type'] = t[4].strip()
                    t_data.append(temp_data)
                    a += 1
                else:
                    incomplete_generation = True
                    if fix_missing is True:
                        print('Hi')
                        # print(t[1].strip())
                        temp_data = info.copy()
                        temp_data['id'] = counter
                        temp_data['uuid'] = f_name
                        temp_data['place_name'] = 'Home'
                        temp_data['arrival_time'] = \
                            t_data[-1]['departure_time']
                        temp_data['departure_time'] = '11:59 PM'
                        temp_data['loc_type'] = '1'
                        t_data.append(temp_data)
                        a += 1
                    else:
                        break
        if incomplete_generation:
            incomplete_generations += 1
            # print(f)
            if fix_missing is True:
                data.extend(t_data)
                counter += 1
            else:
                continue
        else:
            counter += 1
            data.extend(t_data)
        # except Exception as e:
        #     e
    print(f'Complete generations: {counter}')
    print(f'Incomplete generations: {incomplete_generations}')
    print(f'Table not found: {table_not_found}')
    return data


def process_outputs_agent_palm(
    folder_loc='outputs/',
    fix_missing=False
):
    files = os.listdir(folder_loc)
    data = []
    counter = 0
    incomplete_generations = 0
    table_not_found = 0
    fix_missing = False
    for f in files:
        f_name = f.split('.')[0]
        t_data = []
        incomplete_generation = False
        if f.split('.')[-1] != 'json':
            continue
        answers = json.load(open(folder_loc+f))
        info = answers.copy()

        for a in [
            'prompt_survey',
            'result_survey',
            'prompt_analysis',
            'result_analysis',
        ]:
            del info[a]

        ans_output = answers['result_analysis']
        ds = ans_output.split('\n')
        ds = [d.strip() for d in ds]
        a = None
        for i, each in enumerate(ds):
            pattern = re.compile(
                r"\|\s*-+\s*\|\s*-+\s*\|\s*-+\s*\|\s*-+\s*\|",
                re.IGNORECASE
            )
            m = pattern.match(each)
            if m is None:
                pass
            else:
                a = i
                break
        if a is None:
            table_not_found += 1
            continue

        while True:
            if a+1 >= len(ds):
                break
            elif ds[a+1] == '':
                break
            else:
                t = ds[a+1].split('|')
                if len(t) == 6:
                    temp_data = info.copy()
                    temp_data['id'] = counter
                    temp_data['uuid'] = f_name
                    temp_data['place_name'] = t[1].strip()
                    temp_data['arrival_time'] = t[2].strip()
                    temp_data['departure_time'] = t[3].strip()
                    temp_data['loc_type'] = t[4].strip()
                    t_data.append(temp_data)
                    a += 1
                else:
                    incomplete_generation = True
                    if fix_missing is True:
                        print('Hi')
                        # print(t[1].strip())
                        temp_data = info.copy()
                        temp_data['id'] = counter
                        temp_data['uuid'] = f_name
                        temp_data['place_name'] = 'Home'
                        temp_data['arrival_time'] = \
                            t_data[-1]['departure_time']
                        temp_data['departure_time'] = '11:59 PM'
                        temp_data['loc_type'] = '1'
                        t_data.append(temp_data)
                        a += 1
                    else:
                        break
        if incomplete_generation:
            incomplete_generations += 1
            # print(f)
            if fix_missing is True:
                data.extend(t_data)
                counter += 1
            else:
                continue
        else:
            counter += 1
            data.extend(t_data)
        # except Exception as e:
        #     e
    print(f'Complete generations: {counter}')
    print(f'Incomplete generations: {incomplete_generations}')
    print(f'Table not found: {table_not_found}')
    return data


def process_outputs_completion_gpt4(
    folder_loc='outputs/',
    fix_missing=False
):
    files = os.listdir(folder_loc)
    data = []
    counter = 0
    incomplete_generations = 0
    table_not_found = 0
    fix_missing = False
    for f in files:
        f_name = f.split('.')[0]
        t_data = []
        incomplete_generation = False
        if f.split('.')[-1] != 'json':
            continue
        answers = json.load(open(folder_loc+f))
        info = answers.copy()

        for a in ['ans_output', 'input']:
            del info[a]

        ans_output = answers['ans_output']
        ds = ans_output.split('\n')
        ds = [d.strip() for d in ds]
        a = None
        for i, each in enumerate(ds):
            pattern = re.compile(
                r"\|\s*-+\s*\|\s*-+\s*\|\s*-+\s*\|\s*-+\s*\|",
                re.IGNORECASE
            )
            m = pattern.match(each)
            if m is None:
                pass
            else:
                a = i
                break
        if a is None:
            table_not_found += 1
            continue

        while True:
            if a+1 >= len(ds):
                break
            elif ds[a+1] == '':
                break
            else:
                t = ds[a+1].split('|')
                if len(t) == 6:
                    temp_data = info.copy()
                    temp_data['id'] = counter
                    temp_data['uuid'] = f_name
                    temp_data['place_name'] = t[1].strip()
                    temp_data['arrival_time'] = t[2].strip()
                    temp_data['departure_time'] = t[3].strip()
                    temp_data['loc_type'] = t[4].strip()
                    t_data.append(temp_data)
                    a += 1
                else:
                    incomplete_generation = True
                    if fix_missing is True:
                        print('Hi')
                        # print(t[1].strip())
                        temp_data = info.copy()
                        temp_data['id'] = counter
                        temp_data['uuid'] = f_name
                        temp_data['place_name'] = 'Home'
                        temp_data['arrival_time'] = \
                            t_data[-1]['departure_time']
                        temp_data['departure_time'] = '11:59 PM'
                        temp_data['loc_type'] = '1'
                        t_data.append(temp_data)
                        a += 1
                    else:
                        break
        if incomplete_generation:
            incomplete_generations += 1
            # print(f)
            if fix_missing is True:
                data.extend(t_data)
                counter += 1
            else:
                continue
        else:
            counter += 1
            data.extend(t_data)
        # except Exception as e:
        #     e
    print(f'Complete generations: {counter}')
    print(f'Incomplete generations: {incomplete_generations}')
    print(f'Table not found: {table_not_found}')
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='process_outputs'
    )
    parser.add_argument(
        '--type',
        dest='type',
        type=str,
        choices=[
            'survey',
            'completion',
            'completion_palm',
            'agent_palm',
            'completion_gpt4'
        ],
        default='survey'
    )
    parser.add_argument(
        '--in_folder',
        dest='in_folder',
        type=str,
        default='outputs/'
    )

    parser.add_argument(
        '--out_folder',
        dest='out_folder',
        type=str,
        default='outputs_processed/'
    )

    parser.add_argument(
        '--file_name',
        dest='file_name',
        type=str,
        # default='outputs_processed'
        default='outputs_processed'
    )

    parser.add_argument(
        '--fix_missing',
        dest='fix_missing',
        type=bool,
        default=False
    )

    args = parser.parse_args()

    if args.type == 'completion_palm':
        data = process_outputs_completion_palm(
            args.in_folder,
            False
        )
    elif args.type == 'agent_palm':
        data = process_outputs_agent_palm(
            args.in_folder,
            False
        )
    elif args.type == 'completion':
        data = process_outputs_completion(
            args.in_folder,
            True
        )
    elif args.type == 'completion_gpt4':
        data = process_outputs_completion_gpt4(
            args.in_folder,
            False
        )
    else:
        data = process_outputs_new(
            args.in_folder,
            args.fix_missing
        )
    df = pd.DataFrame(data)
    df.to_csv(f'{args.out_folder}/{args.file_name}.csv', index=False)
