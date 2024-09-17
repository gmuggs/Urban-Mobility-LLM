import os
import uuid
import json
import random
from datetime import date
from model_inference import (
    conduct_survey,
    conduct_completion,
    conduct_palm_completion,
    conduct_gpt3_completion,
    conduct_gpt4_completion,
    conduct_llama2_70b_api_completion,
    conduct_completion_llama2_70b,
    conduct_gemini_completion
)
from generate_system_prompt import (
    generate_system_prompt,
    generate_completion_prompt,
    generate_completion_prompt_simplified
)
from sample_population import (
    sample_from_sf_popn,
    sample_from_uk_population,
    sample_date
)
from transformers import LlamaTokenizer
try:
    from vllm import LLM, SamplingParams
except ModuleNotFoundError:
    print("Can not import vllm, will only work with local models")
import google.generativeai as genai
from utils import (
    load_model,
    load_trained_model
)
import argparse
from get_acs_data_files import get_acs_data_files

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Similate a survey using LLMs.'
    )
    parser.add_argument(
        '--local_rank',
    )
    parser.add_argument(
        '--type',
        dest='type',
        type=str,
        choices=[
            'survey',
            'completion'
        ],
        default='completion'
    )
    parser.add_argument(
        '--location',
        dest='location',
        type=str,
        choices=[
            'sf',
            'dc',
            'dfw',
            'minneapolis',
            'la'
        ]
    )

    parser.add_argument(
        '--travel_file',
        dest='travel_file',
        type=str,
        choices=[
            'travel',
            'travel2',
            'travel3'
        ],
        default='travel'
    )

    parser.add_argument(
        '--loc_ub',
        dest='loc_ub',
        type=str,
        choices=[
            'downtown',
            'suburb',
            'none'
        ],
        default='none'
    )

    parser.add_argument(
        '--use_popn_sampling',
        dest='use_popn_sampling',
        type=str,
        choices=[
            'true',
            'false',
        ],
        default='true'
    )

    parser.add_argument(
        '--use_date_sampling',
        dest='use_date_sampling',
        type=str,
        choices=[
            'true',
            'false',
        ],
        default='true'
    )

    parser.add_argument(
        '--use_model',
        dest='use_model',
        type=str,
        choices=[
            'llama',
            'palm',
            'gpt3',
            'gpt4',
            'llama2-70b',
            'llama2-70b-api',
            'gemini',
            'llama-2-trained'
        ],
        default='llama'
    )

    parser.add_argument(
        '--out_folder',
        dest='out_folder',
        type=str,
        default='outputs'
    )

    parser.add_argument(
        '--fix_bias',
        dest='fix_bias',
        type=str,
        default='false',
        choices=[
            'false',
            '1',
            '2',
            'covid'
        ]
    )
    parser.add_argument(
        '--year',
        dest='year',
        type=str,
        default='2021',
        choices=[
            '2017',
        ]
    )
    parser.add_argument(
        '--trained_model_epoch',
        dest='trained_model_epoch',
        type=int,
        default=1,
        choices=[
            1,
            3,
            5,
            10,
            20
        ]
    )
    parser.add_argument(
        '--trained_db_size',
        dest='trained_sb_size',
        type=str,
        default=None,
        choices=[
            None,
            '1000',
            '10000',
            '10000-exclude-inf-cities',
        ]
    )
    args = parser.parse_args()
    if args.location == 'fairfax':
        location = "Fairfax, VA"
    elif args.location == 'dc':
        location = "dc"
        location2 = "Washington-Arlington-Alexandria, DC-VA-MD-WV Metro Area"
    elif args.location == 'sf':
        location = "San Francisco, CA"
        if args.year == '2021':
            location2 = "San Francisco, CA"
        elif args.year == '2017':
            location2 = "San Francisco-Oakland-Hayward, CA Metro Area"
    elif args.location == 'dfw':
        location = "dfw"
        location2 = "Dallas-Fort Worth-Arlington, TX Metro Area"
    elif args.location == 'minneapolis':
        location = "minneapolis"
        location2 = "Minneapolis-St. Paul-Bloomington, MN-WI Metro Area"
    elif args.location == 'la':
        location = "la"
        location2 = "Los Angeles-Long Beach-Anaheim, CA Metro Area"
    elif args.location == 'chicago':
        location = "Chicago, IL"
    elif args.location == 'houston':
        location = "Houston, TX"
        location2 = "Houston, TX"
    elif args.location == 'pittsburgh':
        location = "Pittsburgh, PA"
        location2 = "Pittsburgh, PA"
    elif args.location == 'ok_city':
        location = "Oklahoma City, OK"
        location2 = "Oklahoma City, OK"
    elif args.location == 'baltimore':
        location = "Baltimore, MD"
        location2 = "Baltimore, MD"
    elif args.location == 'london':
        location = "Greater London Area (UK)"

    if location == "Greater London Area (UK)":
        TS008_loc, RM121_loc, TS021_loc, TS066_loc, RM074_loc, _ =\
            get_acs_data_files(location)
    else:
        B01001_loc, S1401_loc, S2301_loc, S2401_loc, S1201_loc, S1101_loc = \
            get_acs_data_files(location, year=int(args.year))
    start_date = date(2016, 4, 19)
    end_date = date(2017, 4, 25)

    if args.type == 'completion':
        model_name = 'meta-llama/Llama-2-70b-hf'
    else:
        model_name = 'meta-llama/Llama-2-70b-chat-hf'

    quantization = True
    if args.use_model == 'palm':
        genai.configure(api_key=open("palm_api_key2").read().strip())
        models = [
            m
            for m in genai.list_models()
            if 'generateText' in m.supported_generation_methods
        ]
        model = models[0].name
    elif args.use_model == 'gemini':
        rand_int = random.randint(1, 2)
        if rand_int == 1:
            genai.configure(api_key=open("palm_api_key").read().strip())
        else:
            genai.configure(api_key=open("palm_api_key2").read().strip())
        model = genai.GenerativeModel('gemini-pro')
    elif args.use_model == 'gpt3':
        model = 'gpt-3.5-turbo'
    elif args.use_model == 'gpt4':
        model = 'gpt-4-turbo-preview'
    elif args.use_model == 'llama2-70b':
        model_name = 'meta-llama/Llama-2-70b-hf'
        from huggingface_hub import login
        login(token=open('access_token').read())
        sampling_params = SamplingParams(
            temperature=1,
            top_p=1,
            top_k=50,
            max_tokens=512,
            repetition_penalty=1.0,
            length_penalty=1.0,
        )
        llm = LLM(model_name, tensor_parallel_size=2)
    elif args.use_model == 'llama2-70b-api':
        model = 'meta/llama-2-70b'
    elif args.use_model == 'llama-2-trained':
        model = load_trained_model(
            model_name,
            epoch=args.trained_model_epoch,
            db_size=args.trained_sb_size
        )
        print(model)
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
    else:
        model = load_model(model_name, quantization=quantization)
        if not quantization:
            model.to("cuda:0")
        tokenizer = LlamaTokenizer.from_pretrained(model_name)

    for i in range(500):
        if args.use_date_sampling == 'true':
            (
                survey_date,
                survey_date_weekday
            ) = sample_date(start_date, end_date)
        else:
            survey_date = date(2017, 4, 25)
            survey_date_weekday = 'Tuesday'

        if args.use_popn_sampling == 'true':
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
                sample_occupation = None
                sample_household_type = None
                sample_own_child_under_18 = None
                sample_own_child_under_18_type = None
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
                    year=int(args.year)
                )
        else:
            sample_sex = 'male'
            sample_age_group = '45 to 49 years'
            sample_age = 49
            sample_race = 'White alone'
            sample_enrolled_in_school = False
            sample_participation_in_labor_force = True
            sample_employed = True
            sample_occupation = \
                'Educational instruction, and library occupations'
            sample_marital_status = 'never married'
            sample_household_type = 'non family household'
            sample_own_child_under_18 = None
            sample_own_child_under_18_type = None

        if args.type == 'completion':
            if location == "Greater London Area (UK)":
                prompt = generate_completion_prompt_simplified(
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
                )
            else:
                if args.year == '2021':
                    prompt = generate_completion_prompt(
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
                        args.loc_ub
                    )
                if args.year == '2017':
                    prompt = generate_completion_prompt(
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
                        location2,
                        survey_date,
                        survey_date_weekday,
                        args.loc_ub
                    )
            if args.fix_bias == '1':
                additional_text = (
                    'This survey aims for realistic representations of '
                    'individual trips. Please be mindful of potential biases '
                    'observed in previous responses: adjust for a 7.5% '
                    'overestimation of work-based locations and a 5% '
                    'underestimation of shopping destinations. Additionally, '
                    'consider a 2.25% underestimation in social activities.'
                )
                modified_prompt = (
                    f"{additional_text} Now, complete the following: {prompt}"
                )
                prompt = modified_prompt
            elif args.fix_bias == '2':
                additional_text = (
                    'This survey seeks realistic representations of individual'
                    ' trips. Consider and correct potential biases observed in'
                    ' prior responses, such as overestimations in work-based'
                    ' locations and underestimations in shopping destinations.'
                )
                modified_prompt = (
                    f"{additional_text} Now, complete the following: {prompt}"
                )
                prompt = modified_prompt
            elif args.fix_bias == 'covid':
                additional_text = (
                    'This survey was done during the COVID-19 pandemic. The '
                    'COVID-19 pandemic drastically altered daily commuting '
                    'patterns in the USA. Remote work became widespread, '
                    'leading to a decline in traditional workplace commuting '
                    'and a reduction in public transit ridership. Active '
                    'transportation modes saw increased usage, while traffic '
                    'patterns shifted, with some areas experiencing reduced '
                    'congestion during peak hours.'
                )
                modified_prompt = (
                    f"{additional_text} Now, complete the following: {prompt}"
                )
                prompt = modified_prompt
        else:
            prompt = generate_system_prompt(
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
                args.loc_ub
            )
        if sample_age < 16:
            continue

        if args.type == 'completion':
            if args.use_model == 'palm':
                answers = conduct_palm_completion(
                    model,
                    prompt,
                )
            elif args.use_model == 'gpt3':
                answers = conduct_gpt3_completion(
                    model,
                    prompt,
                )
            elif args.use_model == 'gpt4':
                answers = conduct_gpt4_completion(
                    model,
                    prompt,
                )
            elif args.use_model == 'llama2-70b':
                answers = conduct_completion_llama2_70b(
                    llm,
                    sampling_params=sampling_params,
                    prompt_file=prompt,
                )
            elif args.use_model == 'llama2-70b-api':
                answers = conduct_llama2_70b_api_completion(
                    model,
                    prompt,
                )
            elif args.use_model == 'gemini':
                answers = conduct_gemini_completion(
                    model,
                    prompt,
                )
            else:
                answers = conduct_completion(
                    model,
                    tokenizer,
                    prompt_file=prompt,
                )
            answers.update({
                'sex': sample_sex,
                'age_group': sample_age_group,
                'age': sample_age,
                'race': sample_race,
                'enrolled_in_school': sample_enrolled_in_school,
                'participation_in_labor_force':
                    sample_participation_in_labor_force,
                'employed': sample_employed,
                'occupation': sample_occupation,
                'marital_status': sample_marital_status,
                'own_child_under_18': sample_own_child_under_18,
                'own_child_under_18_type': sample_own_child_under_18_type,
                'location': location2,
                'survey_date': str(survey_date),
                'survey_date_weekday': survey_date_weekday,
                'loc_ub': args.loc_ub,
                'model': args.use_model,
            })
            json.dump(
                answers,
                open(
                    f'{args.out_folder}/completions_{uuid.uuid4()}.json',
                    'w+'
                )
            )
        else:
            absolute_path = os.path.dirname(__file__)
            relative_path = f"surveys/{args.travel_file}.json"
            full_path = os.path.join(absolute_path, relative_path)
            travel_survey = json.load(open(full_path, 'r'))

            answers = conduct_survey(
                model,
                tokenizer,
                prompt_file=prompt,
                survey_questions=travel_survey,
            )
            answers.append({
                'sex': sample_sex,
                'age_group': sample_age_group,
                'age': sample_age,
                'race': sample_race,
                'enrolled_in_school': sample_enrolled_in_school,
                'participation_in_labor_force':
                    sample_participation_in_labor_force,
                'employed': sample_employed,
                'occupation': sample_occupation,
                'marital_status': sample_marital_status,
                'own_child_under_18': sample_own_child_under_18,
                'own_child_under_18_type': sample_own_child_under_18_type,
                'location': location,
                'survey_date': str(survey_date),
                'survey_date_weekday': survey_date_weekday,
                'loc_ub': args.loc_ub,
            })
            json.dump(
                answers,
                open(
                    f'{args.out_folder}/survey_answers_{uuid.uuid4()}.json',
                    'w+'
                )
            )
