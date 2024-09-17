def get_age_range(age: int):
    if (age >= 5) and (age <= 15):
        return "5-15"
    elif (age >= 16) and (age <= 17):
        return "16-17"
    elif (age >= 18) and (age <= 24):
        return "18-24"
    elif (age >= 25) and (age <= 34):
        return "25-34"
    elif (age >= 35) and (age <= 54):
        return "35-54"
    elif (age >= 55) and (age <= 64):
        return "55-64"
    elif (age >= 65) and (age <= 75):
        return "65-75"
    else:
        return "76 or older"


def get_primary_activity(primary_activity: str):
    if primary_activity == "work":
        return "You are a working person."
    elif primary_activity == "retired":
        return "You have retired."
    elif primary_activity == "home_maker":
        return "You are a home maker."
    elif primary_activity == "school":
        return "You are a student."
    elif primary_activity == "unemployed":
        return "You are unemployed."
    elif primary_activity == "temp":
        return "You are temporarily absent from a job or business."
    else:
        return "You are not in the labor force."


def generate_system_prompt_old(
        gender: str,  # [male, female, ]
        age: int,
        location: str,
        primary_activity: str,
        date: str,
):
    content = f"You are a {gender}."
    content += f" You age lies in the range of {get_age_range(age)}."
    content += f" You live in {location}."
    content += get_primary_activity(primary_activity)
    content += (
        "You have been selected for a travel survey and have "
        f"recorded your travel logs for {date}."
    )
    prompt = [
        [
            {
                "role": "system",
                "content": content
            }
        ],
    ]

    return prompt


def generate_system_prompt(
    sex,
    age_group,
    age,
    race,
    enrolled_in_school,
    participation_in_labor_force,
    employed,
    occupation,
    marital_status,
    sample_household_type,
    own_child_under_18,
    own_child_under_18_type,
    location,
    survey_date,
    survey_date_weekday,
    loc_ub
):

    content = (
        f"You are a person who identifies as {sex}, "
        f"with an age of {age}. "
    )
    content += f"Your racial background is '{race}'. "
    if enrolled_in_school is True:
        content += "Currently, you are enrolled in school "
    else:
        content += "Currently, you are not enrolled in school "
    if participation_in_labor_force is True:
        content += "and you are participating in the labor force. "
        if employed is True:
            content += "You are employed "
            content += f"and working in the '{occupation}' field. "
        else:
            content += "You are not employed. "
    else:
        content += "and you are not participating in the labor force. "
    if marital_status == "never married":
        content += (
            "Regarding your marital status, you have never been married, "
        )
    else:
        content += f"Regarding your marital status, you are {marital_status}, "
    content += f"and you live in a {sample_household_type}. "
    if own_child_under_18 is True:
        if own_child_under_18_type == "under 6 years only":
            content += (
                "Additionally, your household has child(ren) "
                "under the age of 6 only. "
            )
        elif own_child_under_18_type == "under 6 years and 6 to 17 years":
            content += (
                "Additionally, your household has child(ren) "
                "under the age of 18. "
            )
        elif own_child_under_18_type == "6 to 17 years only":
            content += (
                "Additionally, your household has child(ren) "
                "between the age of 6 to 17 years. "
            )
        # content += "Additionally, your household has child(ren) under 18, "
        # content += (
        #     f"and the type of child(ren) is {own_child_under_18_type}. "
        # )
    if loc_ub == 'none':
        content += f" You live in {location}. "
    elif loc_ub == 'downtown':
        content += f" You live in the downtown area of {location}. "
    elif loc_ub == 'suburb':
        content += f" You live in the suburban area of {location}. "
    else:
        content += f" You live in {location}. "

    content += (
        "You have been selected for a travel survey and have "
        f"recorded your travel logs for {survey_date} "
        f"which is a {survey_date_weekday}."
    )
    prompt = [
        [
            {
                "role": "system",
                "content": content
            }
        ],
    ]

    return prompt


def generate_completion_prompt(
    sex,
    age_group,
    age,
    race,
    enrolled_in_school,
    participation_in_labor_force,
    employed,
    occupation,
    marital_status,
    sample_household_type,
    own_child_under_18,
    own_child_under_18_type,
    location,
    survey_date,
    survey_date_weekday,
    loc_ub
):
    content = (
        f"The individual is a {age}-year-old {sex} "
        f"who's racial background is '{race}'. "
    )
    if enrolled_in_school is True:
        if sex == 'male':
            content += "Currently, he is enrolled in school "
        elif sex == 'female':
            content += "Currently, she is enrolled in school "
    else:
        if sex == 'male':
            content += "Currently, he is not enrolled in school "
        elif sex == 'female':
            content += "Currently, she is not enrolled in school "
    if participation_in_labor_force is True:
        content += "and is participating in the labor force. "
        if employed is True:
            if sex == 'male':
                content += "He is employed "
                content += f"and working in the '{occupation}' field. "
            elif sex == 'female':
                content += "She is employed "
                content += f"and working in the '{occupation}' field. "
        else:
            if sex == 'male':
                content += "He is not employed. "
            elif sex == 'female':
                content += "She is not employed. "
    else:
        content += "and is not participating in the labor force. "
    if marital_status == "never married":
        if sex == 'male':
            content += (
                "Regarding his marital status, he has never been married, "
            )
        elif sex == 'female':
            content += (
                "Regarding her marital status, she has never been married, "
            )
    else:
        if sex == 'male':
            content += (
                f"Regarding his marital status, he is {marital_status}, "
            )
        elif sex == 'female':
            content += (
                f"Regarding her marital status, she is {marital_status}, "
            )
    content += f"and lives in a {sample_household_type}. "
    if own_child_under_18 is True:
        if own_child_under_18_type == "under 6 years only":
            if sex == 'male':
                content += (
                    "Additionally, his household has child(ren) "
                    "under the age of 6 only. "
                )
            elif sex == 'female':
                content += (
                    "Additionally, her household has child(ren) "
                    "under the age of 6 only. "
                )
        elif own_child_under_18_type == "under 6 years and 6 to 17 years":
            if sex == 'male':
                content += (
                    "Additionally, his household has child(ren) "
                    "under the age of 18. "
                )
            elif sex == 'female':
                content += (
                    "Additionally, her household has child(ren) "
                    "under the age of 18. "
                )
        elif own_child_under_18_type == "6 to 17 years only":
            if sex == 'male':
                content += (
                    "Additionally, his household has child(ren) "
                    "between the age of 6 to 17 years. "
                )
            elif sex == 'female':
                content += (
                    "Additionally, her household has child(ren) "
                    "between the age of 6 to 17 years. "
                )
    if loc_ub == 'none':
        if sex == 'male':
            content += f" He lives in {location}. "
        elif sex == 'female':
            content += f" She lives in {location}. "
    elif loc_ub == 'downtown':
        if sex == 'male':
            content += f" He lives in the downtown area of {location}. "
        elif sex == 'female':
            content += f" She lives in the downtown area of {location}. "
    elif loc_ub == 'suburb':
        if sex == 'male':
            content += f" He lives in the suburban area of {location}. "
        elif sex == 'female':
            content += f" She lives in the suburban area of {location}. "
    else:
        if sex == 'male':
            content += f" He lives in {location}. "
        elif sex == 'female':
            content += f" She lives in {location}. "

    if sex == 'male':
        content += (
            "He has been selected for a travel survey and have "
            f"recorded his travel logs for {survey_date} "
            f"which is a {survey_date_weekday}. \n"
            "He was asked to provide a list of all the places he visited on "
            "his travel date, "
        )
    elif sex == 'female':
        content += (
            "She has been selected for a travel survey and have "
            f"recorded her travel logs for {survey_date} "
            f"which is a {survey_date_weekday}. \n"
            "She was asked to provide a list of all the places she visited on "
            "her travel date, "
        )
    content += (
        "including the exact times of arrival and departure, and the location "
        "type. The table format provided was as follows: \n\n"
        "| Place Visited           | Arrival Time    | Departure Time  | "
        "Location Type   |\n|-------------------------|-----------------|-----"
        "------------|-----------------|\n| [Place Name]            | [HH:MM "
        "AM/PM]   | [HH:MM AM/PM]   | [Location Type] |\n| [Place Name]       "
        "     | [HH:MM AM/PM]   | [HH:MM AM/PM]   | [Location Type] |\n| ...  "
        "                   | ...             | ...             | ...         "
        "    |\n\n"
    )
    if sex == 'male':
        content += (
            "He was instructed to fill in each row with the relevant "
            "information for each place he visited on the specified date. If "
            "he visited the same place multiple times on the same date, he was"
            " advised to add a separate row for each visit to that place.\n\n"
            "He was reminded of the following:\n\n"
        )
    elif sex == 'female':
        content += (
            "She was instructed to fill in each row with the relevant "
            "information for each place she visited on the specified date. If "
            "she visited the same place multiple times on the same date, she "
            "was advised to add a separate row for each visit to that place. "
            "\n\nShe was reminded of the following:\n\n"
        )
    content += (
        "1. Ensure that 'Home' is included in the list if it was part of "
        " travel activities on the specified date.\n"
    )
    if sex == 'male':
        content += (
            "2. He was asked to input the exact arrival and departure time as"
            " he noted in his travel diary.\n3. He was advised to carefully "
            "enter the times, as gaps between the departure time of the "
            "previous place and the arrival time of the current place indicate"
            " the time taken to arrive at the current location."
            "\n\nFor the 'Location Type,' he was instructed to use the "
            "corresponding code from the provided list:\n\n"
        )
    elif sex == 'female':
        content += (
            "2. She was asked to input the exact arrival and departure time as"
            " she noted in her travel diary.\n3. She was advised to carefully "
            "enter the times, as gaps between the departure time of the "
            "previous place and the arrival time of the current place indicate"
            " the time taken to arrive at the current location."
            "\n\nFor the 'Location Type,' she was instructed to use the "
            "corresponding code from the provided list:\n\n"
        )
    content += (
        "1: Regular home activities (chores, sleep)\n2: Work from home (paid)"
        "\n3: Work\n4: Work-related meeting / trip\n5: Volunteer activities "
        "(not paid)\n6: Drop off / pick up someone\n7: Change type of "
        "transportation\n8: Attend school as a student\n9: Attend child care"
        "\n10: Attend adult care\n11: Buy goods (groceries, clothes, "
        "appliances, gas)\n12: Buy services (dry cleaners, banking, service a "
        "car, etc)\n13: Buy meals (go out for a meal, snack, carry-out)\n14: "
        "Other general errands (post office, library)\n15: Recreational "
        "activities (visit parks, movies, bars, etc)\n16: Exercise (go for a "
        "jog, walk, walk the dog, go to the gym, etc)\n17: Visit friends or "
        "relatives\n18: Health care visit (medical, dental, therapy)\n19: "
        "Religious or other community activities\n97: Something else\n\n"
    )
    if sex == 'male':
        content += (
            "The table he created is as follows:\n"
        )
    elif sex == 'female':
        content += (
            "The table she created is as follows:\n"
        )

    return content


def generate_agent_prompt_survey(
    sex,
    age_group,
    age,
    race,
    enrolled_in_school,
    participation_in_labor_force,
    employed,
    occupation,
    marital_status,
    household_type,
    own_child_under_18,
    own_child_under_18_type,
    location,
    survey_date,
    survey_date_weekday,
    loc_ub='none'
):
    content = (
        f"Person A is a {age}-year-old {sex} "
        f"who's racial background is '{race}'. "
    )
    if enrolled_in_school is True:
        if sex == 'male':
            content += "Currently, he is enrolled in school "
        elif sex == 'female':
            content += "Currently, she is enrolled in school "
    else:
        if sex == 'male':
            content += "Currently, he is not enrolled in school "
        elif sex == 'female':
            content += "Currently, she is not enrolled in school "
    if participation_in_labor_force is True:
        content += "and is participating in the labor force. "
        if employed is True:
            if sex == 'male':
                content += "He is employed "
                content += f"and working in the '{occupation}' field. "
            elif sex == 'female':
                content += "She is employed "
                content += f"and working in the '{occupation}' field. "
        else:
            if sex == 'male':
                content += "He is not employed. "
            elif sex == 'female':
                content += "She is not employed. "
    else:
        content += "and is not participating in the labor force. "
    if marital_status == "never married":
        if sex == 'male':
            content += (
                "Regarding his marital status, he has never been married, "
            )
        elif sex == 'female':
            content += (
                "Regarding her marital status, she has never been married, "
            )
    else:
        if sex == 'male':
            content += (
                f"Regarding his marital status, he is {marital_status}, "
            )
        elif sex == 'female':
            content += (
                f"Regarding her marital status, she is {marital_status}, "
            )
    content += f"and lives in a {household_type}. "
    if own_child_under_18 is True:
        if own_child_under_18_type == "under 6 years only":
            if sex == 'male':
                content += (
                    "Additionally, his household has child(ren) "
                    "under the age of 6 only. "
                )
            elif sex == 'female':
                content += (
                    "Additionally, her household has child(ren) "
                    "under the age of 6 only. "
                )
        elif own_child_under_18_type == "under 6 years and 6 to 17 years":
            if sex == 'male':
                content += (
                    "Additionally, his household has child(ren) "
                    "under the age of 18. "
                )
            elif sex == 'female':
                content += (
                    "Additionally, her household has child(ren) "
                    "under the age of 18. "
                )
        elif own_child_under_18_type == "6 to 17 years only":
            if sex == 'male':
                content += (
                    "Additionally, his household has child(ren) "
                    "between the age of 6 to 17 years. "
                )
            elif sex == 'female':
                content += (
                    "Additionally, her household has child(ren) "
                    "between the age of 6 to 17 years. "
                )
    if sex == 'male':
        content += f" He lives in {location}. "
    elif sex == 'female':
        content += f" She lives in {location}. "

    if sex == 'male':
        content += (
            "He has been selected for a travel survey and have "
            f"recorded his travel logs for {survey_date} "
            f"which is a {survey_date_weekday}. \n"
        )
    elif sex == 'female':
        content += (
            "She has been selected for a travel survey and have "
            f"recorded her travel logs for {survey_date} "
            f"which is a {survey_date_weekday}. \n"
        )

    content += (
        "\n"
        "Person B is an enumerator who is conducting the travel survey of "
        "Person A. Person B knows that Person A has been selected for a travel"
        " survey and have  recorded their travel logs for the specified date. "
    )

    content += (
        "The instruction provided to Person B is as follows:\n"
        "1. Start the survey by telling person A that you are conducting the "
        "travel survey and that you will be asking them about their travel "
        "activities on the specified date and ask if they have recored their "
        "travel logs for that specified day or not.\n"
        "2. Then tell them that a place is any location you go to, no matter "
        "how long you are there and if you visit same place twice, you should "
        "count it as two separate places.\n"
        "3. Then start the survey by asking the first place they were at the "
        "start of the day and when they left.\n"
        "4. For all subsequent place, inquire about the place name, arrival "
        "time, and departure time.\n"
        "5. Always end the survey with 'Thank you for participatin in our "
        "survey.'\n"
    )

    content += (
        "\n The transcript of the conversation between Person A and Person B "
        "is as follows:\n"
    )

    return content


def genereate_agent_prompt_analysis(
    conversasation
):
    prompt = (
        "Person C is a survey analyst who is analyzing the travel survey of "
        "Person A. Person C is presented with a conversation between Person A "
        "and Person B, where Person A is the person being surveyed and Person "
        "B is the enumerator who is conducting the survey. Person C is tasked "
        "with analyzing the conversation and creating a travel log table. The "
        "table format per Person C to create the travel log table is as "
        "follows: \n\n"
        "| Place Visited           | Arrival Time    | Departure Time  | "
        "Location Type   |\n|-------------------------|-----------------|-----"
        "------------|-----------------|\n| [Place Name]            | [HH:MM "
        "AM/PM]   | [HH:MM AM/PM]   | [Location Type] |\n| [Place Name]       "
        "     | [HH:MM AM/PM]   | [HH:MM AM/PM]   | [Location Type] |\n| ...  "
        "                   | ...             | ...             | ...         "
        "    |\n\n"
        "For the 'Location Type,' Person C is instructed to think critically "
        "and use the corresponding code from the provided list to provide the "
        "accurate location type:\n\n"
        "1: Regular home activities (chores, sleep)\n2: Work from home (paid)"
        "\n3: Work\n4: Work-related meeting / trip\n5: Volunteer activities "
        "(not paid)\n6: Drop off / pick up someone\n7: Change type of "
        "transportation\n8: Attend school as a student\n9: Attend child care"
        "\n10: Attend adult care\n11: Buy goods (groceries, clothes, "
        "appliances, gas)\n12: Buy services (dry cleaners, banking, service a "
        "car, etc)\n13: Buy meals (go out for a meal, snack, carry-out)\n14: "
        "Other general errands (post office, library)\n15: Recreational "
        "activities (visit parks, movies, bars, etc)\n16: Exercise (go for a "
        "jog, walk, walk the dog, go to the gym, etc)\n17: Visit friends or "
        "relatives\n18: Health care visit (medical, dental, therapy)\n19: "
        "Religious or other community activities\n97: Something else\n\n"
        "Person C is also instructed to make sure that the table is complete "
        "and there are no any missing entries for any columns. If a table is "
        "not complete, Person C is instructed to use his/her intution to fill "
        "up the table. For example, if there is missing departure time for the"
        " last entry, Person C can fill up the departure time as 11:59 PM. "
        "Person C also has to make sure that the arrival time for first entry "
        "start at 00:01 AM and the departure time for the last entry end at "
        "11:59 PM. The transcript of the conversation between Person A and "
        "Person B is as follows:\n"
        f"{conversasation}"
        "\n\n After some analysis, Person C created the travel log table as "
        "follows:\n"
    )
    return prompt


def generate_completion_prompt_simplified(
    sex,
    age,
    race,
    enrolled_in_school,
    participation_in_labor_force,
    employed,
    marital_status,
    location,
    survey_date,
    survey_date_weekday,
):
    content = (
        f"The individual is a {age}-year-old {sex} "
        f"who's racial background is '{race}'. "
    )
    if enrolled_in_school is True:
        if sex == 'male':
            content += "Currently, he is enrolled in school "
        elif sex == 'female':
            content += "Currently, she is enrolled in school "
    else:
        if sex == 'male':
            content += "Currently, he is not enrolled in school "
        elif sex == 'female':
            content += "Currently, she is not enrolled in school "
    if participation_in_labor_force is True:
        content += "and is participating in the labor force. "
        if employed is True:
            if sex == 'male':
                content += "He is employed. "
            elif sex == 'female':
                content += "She is employed. "
        else:
            if sex == 'male':
                content += "He is not employed. "
            elif sex == 'female':
                content += "She is not employed. "
    else:
        content += "and is not participating in the labor force. "
    if marital_status == "never married":
        if sex == 'male':
            content += (
                "Regarding his marital status, he has never been married. "
            )
        elif sex == 'female':
            content += (
                "Regarding her marital status, she has never been married. "
            )
    else:
        if sex == 'male':
            content += (
                f"Regarding his marital status, he is '{marital_status}'. "
            )
        elif sex == 'female':
            content += (
                f"Regarding her marital status, she is '{marital_status}'. "
            )
    if sex == 'male':
        content += f"He lives in {location}. "
    elif sex == 'female':
        content += f"She lives in {location}. "

    if sex == 'male':
        content += (
            "He has been selected for a travel survey and have "
            f"recorded his travel logs for {survey_date} "
            f"which is a {survey_date_weekday}. \n"
            "He was asked to provide a list of all the places he visited on "
            "his travel date, "
        )
    elif sex == 'female':
        content += (
            "She has been selected for a travel survey and have "
            f"recorded her travel logs for {survey_date} "
            f"which is a {survey_date_weekday}. \n"
            "She was asked to provide a list of all the places she visited on "
            "her travel date, "
        )
    content += (
        "including the exact times of arrival and departure, and the location "
        "type. The table format provided was as follows: \n\n"
        "| Place Visited           | Arrival Time    | Departure Time  | "
        "Location Type   |\n|-------------------------|-----------------|-----"
        "------------|-----------------|\n| [Place Name]            | [HH:MM "
        "AM/PM]   | [HH:MM AM/PM]   | [Location Type] |\n| [Place Name]       "
        "     | [HH:MM AM/PM]   | [HH:MM AM/PM]   | [Location Type] |\n| ...  "
        "                   | ...             | ...             | ...         "
        "    |\n\n"
    )
    if sex == 'male':
        content += (
            "He was instructed to fill in each row with the relevant "
            "information for each place he visited on the specified date. If "
            "he visited the same place multiple times on the same date, he was"
            " advised to add a separate row for each visit to that place.\n\n"
            "He was reminded of the following:\n\n"
        )
    elif sex == 'female':
        content += (
            "She was instructed to fill in each row with the relevant "
            "information for each place she visited on the specified date. If "
            "she visited the same place multiple times on the same date, she "
            "was advised to add a separate row for each visit to that place. "
            "\n\nShe was reminded of the following:\n\n"
        )
    content += (
        "1. Ensure that 'Home' is included in the list if it was part of "
        " travel activities on the specified date.\n"
    )
    if sex == 'male':
        content += (
            "2. He was asked to input the exact arrival and departure time as"
            " he noted in his travel diary.\n3. He was advised to carefully "
            "enter the times, as gaps between the departure time of the "
            "previous place and the arrival time of the current place indicate"
            " the time taken to arrive at the current location."
            "\n\nFor the 'Location Type,' he was instructed to use the "
            "corresponding code from the provided list:\n\n"
        )
    elif sex == 'female':
        content += (
            "2. She was asked to input the exact arrival and departure time as"
            " she noted in her travel diary.\n3. She was advised to carefully "
            "enter the times, as gaps between the departure time of the "
            "previous place and the arrival time of the current place indicate"
            " the time taken to arrive at the current location."
            "\n\nFor the 'Location Type,' she was instructed to use the "
            "corresponding code from the provided list:\n\n"
        )
    content += (
        "1: Regular home activities (chores, sleep)\n2: Work from home (paid)"
        "\n3: Work\n4: Work-related meeting / trip\n5: Volunteer activities "
        "(not paid)\n6: Drop off / pick up someone\n7: Change type of "
        "transportation\n8: Attend school as a student\n9: Attend child care"
        "\n10: Attend adult care\n11: Buy goods (groceries, clothes, "
        "appliances, gas)\n12: Buy services (dry cleaners, banking, service a "
        "car, etc)\n13: Buy meals (go out for a meal, snack, carry-out)\n14: "
        "Other general errands (post office, library)\n15: Recreational "
        "activities (visit parks, movies, bars, etc)\n16: Exercise (go for a "
        "jog, walk, walk the dog, go to the gym, etc)\n17: Visit friends or "
        "relatives\n18: Health care visit (medical, dental, therapy)\n19: "
        "Religious or other community activities\n97: Something else\n\n"
    )
    if sex == 'male':
        content += (
            "The table he created is as follows:\n"
        )
    elif sex == 'female':
        content += (
            "The table she created is as follows:\n"
        )

    return content


def generate_completion_prompt_min_info(
    sex,
    age,
    race,
    enrolled_in_school,
    participation_in_labor_force,
    employed,
    marital_status,
    location,
    survey_date,
    survey_date_weekday,
):
    content = (
        f"The individual is a {age}-year-old {sex}. "
    )
    if enrolled_in_school is True:
        if sex == 'male':
            content += "Currently, he is enrolled in school. "
        elif sex == 'female':
            content += "Currently, she is enrolled in school. "
    if employed is True:
        if sex == 'male':
            content += "He is employed. "
        elif sex == 'female':
            content += "She is employed. "
    else:
        if sex == 'male':
            content += "He is not employed. "
        elif sex == 'female':
            content += "She is not employed. "
    if sex == 'male':
        content += f"He lives in {location}. "
    elif sex == 'female':
        content += f"She lives in {location}. "

    if sex == 'male':
        content += (
            "He has been selected for a travel survey and have "
            f"recorded his travel logs for {survey_date} "
            f"which is a {survey_date_weekday}. \n"
            "He was asked to provide a list of all the places he visited on "
            "his travel date, "
        )
    elif sex == 'female':
        content += (
            "She has been selected for a travel survey and have "
            f"recorded her travel logs for {survey_date} "
            f"which is a {survey_date_weekday}. \n"
            "She was asked to provide a list of all the places she visited on "
            "her travel date, "
        )
    content += (
        "including the exact times of arrival and departure, and the location "
        "type. The table format provided was as follows: \n\n"
        "| Place Visited           | Arrival Time    | Departure Time  | "
        "Location Type   |\n|-------------------------|-----------------|-----"
        "------------|-----------------|\n| [Place Name]            | [HH:MM "
        "AM/PM]   | [HH:MM AM/PM]   | [Location Type] |\n| [Place Name]       "
        "     | [HH:MM AM/PM]   | [HH:MM AM/PM]   | [Location Type] |\n| ...  "
        "                   | ...             | ...             | ...         "
        "    |\n\n"
    )
    if sex == 'male':
        content += (
            "He was instructed to fill in each row with the relevant "
            "information for each place he visited on the specified date. If "
            "he visited the same place multiple times on the same date, he was"
            " advised to add a separate row for each visit to that place.\n\n"
            "He was reminded of the following:\n\n"
        )
    elif sex == 'female':
        content += (
            "She was instructed to fill in each row with the relevant "
            "information for each place she visited on the specified date. If "
            "she visited the same place multiple times on the same date, she "
            "was advised to add a separate row for each visit to that place. "
            "\n\nShe was reminded of the following:\n\n"
        )
    content += (
        "1. Ensure that 'Home' is included in the list if it was part of "
        " travel activities on the specified date.\n"
    )
    if sex == 'male':
        content += (
            "2. He was asked to input the exact arrival and departure time as"
            " he noted in his travel diary.\n3. He was advised to carefully "
            "enter the times, as gaps between the departure time of the "
            "previous place and the arrival time of the current place indicate"
            " the time taken to arrive at the current location."
            "\n\nFor the 'Location Type,' he was instructed to use the "
            "corresponding code from the provided list:\n\n"
        )
    elif sex == 'female':
        content += (
            "2. She was asked to input the exact arrival and departure time as"
            " she noted in her travel diary.\n3. She was advised to carefully "
            "enter the times, as gaps between the departure time of the "
            "previous place and the arrival time of the current place indicate"
            " the time taken to arrive at the current location."
            "\n\nFor the 'Location Type,' she was instructed to use the "
            "corresponding code from the provided list:\n\n"
        )
    content += (
        "1: Regular home activities (chores, sleep)\n2: Work from home (paid)"
        "\n3: Work\n4: Work-related meeting / trip\n5: Volunteer activities "
        "(not paid)\n6: Drop off / pick up someone\n7: Change type of "
        "transportation\n8: Attend school as a student\n9: Attend child care"
        "\n10: Attend adult care\n11: Buy goods (groceries, clothes, "
        "appliances, gas)\n12: Buy services (dry cleaners, banking, service a "
        "car, etc)\n13: Buy meals (go out for a meal, snack, carry-out)\n14: "
        "Other general errands (post office, library)\n15: Recreational "
        "activities (visit parks, movies, bars, etc)\n16: Exercise (go for a "
        "jog, walk, walk the dog, go to the gym, etc)\n17: Visit friends or "
        "relatives\n18: Health care visit (medical, dental, therapy)\n19: "
        "Religious or other community activities\n97: Something else\n\n"
    )
    if sex == 'male':
        content += (
            "The table he created is as follows:\n"
        )
    elif sex == 'female':
        content += (
            "The table she created is as follows:\n"
        )

    return content


def generate_train_context(
    sex,
    age,
    race,
    location,
    survey_date,
    survey_date_weekday,
    loc_ub
):
    content = (
        f"The individual is a {age}-year-old {sex} "
        f"who's racial background is '{race}'. "
    )
    if loc_ub == 'none':
        if sex == 'male':
            content += f" He lives in {location}. "
        elif sex == 'female':
            content += f" She lives in {location}. "
    elif loc_ub == 'downtown':
        if sex == 'male':
            content += f" He lives in the downtown area of {location}. "
        elif sex == 'female':
            content += f" She lives in the downtown area of {location}. "
    elif loc_ub == 'suburb':
        if sex == 'male':
            content += f" He lives in the suburban area of {location}. "
        elif sex == 'female':
            content += f" She lives in the suburban area of {location}. "
    else:
        if sex == 'male':
            content += f" He lives in {location}. "
        elif sex == 'female':
            content += f" She lives in {location}. "

    if sex == 'male':
        content += (
            "He has been selected for a travel survey and have "
            f"recorded his travel logs for {survey_date} "
            f"which is a {survey_date_weekday}. \n"
            "He was asked to provide a list of all the places he visited on "
            "his travel date, "
        )
    elif sex == 'female':
        content += (
            "She has been selected for a travel survey and have "
            f"recorded her travel logs for {survey_date} "
            f"which is a {survey_date_weekday}. \n"
            "She was asked to provide a list of all the places she visited on "
            "her travel date, "
        )
    content += (
        "including the exact times of arrival and departure, and the location "
        "type. The table format provided was as follows: \n\n"
        "| Place Visited           | Arrival Time    | Departure Time  | "
        "Location Type   |\n|-------------------------|-----------------|-----"
        "------------|-----------------|\n| [Place Name]            | [HH:MM "
        "AM/PM]   | [HH:MM AM/PM]   | [Location Type] |\n| [Place Name]       "
        "     | [HH:MM AM/PM]   | [HH:MM AM/PM]   | [Location Type] |\n| ...  "
        "                   | ...             | ...             | ...         "
        "    |\n\n"
    )
    if sex == 'male':
        content += (
            "He was instructed to fill in each row with the relevant "
            "information for each place he visited on the specified date. If "
            "he visited the same place multiple times on the same date, he was"
            " advised to add a separate row for each visit to that place.\n\n"
            "He was reminded of the following:\n\n"
        )
    elif sex == 'female':
        content += (
            "She was instructed to fill in each row with the relevant "
            "information for each place she visited on the specified date. If "
            "she visited the same place multiple times on the same date, she "
            "was advised to add a separate row for each visit to that place. "
            "\n\nShe was reminded of the following:\n\n"
        )
    content += (
        "1. Ensure that 'Home' is included in the list if it was part of "
        " travel activities on the specified date.\n"
    )
    if sex == 'male':
        content += (
            "2. He was asked to input the exact arrival and departure time as"
            " he noted in his travel diary.\n3. He was advised to carefully "
            "enter the times, as gaps between the departure time of the "
            "previous place and the arrival time of the current place indicate"
            " the time taken to arrive at the current location."
            "\n\nFor the 'Location Type,' he was instructed to use the "
            "corresponding code from the provided list:\n\n"
        )
    elif sex == 'female':
        content += (
            "2. She was asked to input the exact arrival and departure time as"
            " she noted in her travel diary.\n3. She was advised to carefully "
            "enter the times, as gaps between the departure time of the "
            "previous place and the arrival time of the current place indicate"
            " the time taken to arrive at the current location."
            "\n\nFor the 'Location Type,' she was instructed to use the "
            "corresponding code from the provided list:\n\n"
        )
    content += (
        "1: Regular home activities (chores, sleep)\n2: Work from home (paid)"
        "\n3: Work\n4: Work-related meeting / trip\n5: Volunteer activities "
        "(not paid)\n6: Drop off / pick up someone\n7: Change type of "
        "transportation\n8: Attend school as a student\n9: Attend child care"
        "\n10: Attend adult care\n11: Buy goods (groceries, clothes, "
        "appliances, gas)\n12: Buy services (dry cleaners, banking, service a "
        "car, etc)\n13: Buy meals (go out for a meal, snack, carry-out)\n14: "
        "Other general errands (post office, library)\n15: Recreational "
        "activities (visit parks, movies, bars, etc)\n16: Exercise (go for a "
        "jog, walk, walk the dog, go to the gym, etc)\n17: Visit friends or "
        "relatives\n18: Health care visit (medical, dental, therapy)\n19: "
        "Religious or other community activities\n97: Something else\n\n"
    )
    if sex == 'male':
        content += (
            "The table he created is as follows:\n"
        )
    elif sex == 'female':
        content += (
            "The table she created is as follows:\n"
        )

    return content
