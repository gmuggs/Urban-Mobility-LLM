def get_acs_data_files(location, year=2021):
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
        return TS008_loc, RM121_loc, TS021_loc, TS066_loc, RM074_loc, None
    else:
        if location == "San Francisco, CA":
            if year == 2021:
                base_loc = (
                    'dataset/census_data/san_francisco/'
                    '2021-acs-5-year-estimate/'
                )
                B01001_loc = (
                    base_loc +
                    'ACSDT5YSPT2021.B01001.csv'
                )
                S1401_loc = (
                    base_loc +
                    'ACSST5Y2021.S1401.csv'
                )
                S2301_loc = (
                    base_loc +
                    'ACSST5Y2021.S2301.csv'
                )
                S2401_loc = (
                    base_loc +
                    'ACSST5Y2021.S2401.csv'
                )
                S1201_loc = (
                    base_loc +
                    'ACSST5Y2021.S1201.csv'
                )
                S1101_loc = (
                    base_loc +
                    'ACSST5Y2021.S1101.csv'
                )
            elif year == 2017:
                base_loc = (
                    'dataset/census_data/san_francisco/2017/'
                )
                B01001_loc = (
                    base_loc +
                    'ACSDT5YSPT2015.B01001.csv'
                )
                S1401_loc = (
                    base_loc +
                    'ACSST5Y2017.S1401.csv'
                )
                S2301_loc = (
                    base_loc +
                    'ACSST5Y2017.S2301.csv'
                )
                S2401_loc = (
                    base_loc +
                    'ACSST5Y2017.S2401.csv'
                )
                S1201_loc = (
                    base_loc +
                    'ACSST5Y2017.S1201.csv'
                )
                S1101_loc = (
                    base_loc +
                    'ACSST5Y2017.S1101.csv'
                )
            else:
                raise Exception("Invalid year")
        elif location == "dc":
            base_loc = (
                'dataset/census_data/dc/2017/'
            )
            B01001_loc = (
                base_loc +
                'ACSDT5YSPT2015.B01001.csv'
            )
            S1401_loc = (
                base_loc +
                'ACSST5Y2017.S1401.csv'
            )
            S2301_loc = (
                base_loc +
                'ACSST5Y2017.S2301.csv'
            )
            S2401_loc = (
                base_loc +
                'ACSST5Y2017.S2401.csv'
            )
            S1201_loc = (
                base_loc +
                'ACSST5Y2017.S1201.csv'
            )
            S1101_loc = (
                base_loc +
                'ACSST5Y2017.S1101.csv'
            )
        elif location == "minneapolis":
            base_loc = (
                'dataset/census_data/minneapolis/2017/'
            )
            B01001_loc = (
                base_loc +
                'ACSDT5YSPT2015.B01001.csv'
            )
            S1401_loc = (
                base_loc +
                'ACSST5Y2017.S1401.csv'
            )
            S2301_loc = (
                base_loc +
                'ACSST5Y2017.S2301.csv'
            )
            S2401_loc = (
                base_loc +
                'ACSST5Y2017.S2401.csv'
            )
            S1201_loc = (
                base_loc +
                'ACSST5Y2017.S1201.csv'
            )
            S1101_loc = (
                base_loc +
                'ACSST5Y2017.S1101.csv'
            )
        elif location == "dfw":
            base_loc = (
                'dataset/census_data/dfw/2017/'
            )
            B01001_loc = (
                base_loc +
                'ACSDT5YSPT2015.B01001.csv'
            )
            S1401_loc = (
                base_loc +
                'ACSST5Y2017.S1401.csv'
            )
            S2301_loc = (
                base_loc +
                'ACSST5Y2017.S2301.csv'
            )
            S2401_loc = (
                base_loc +
                'ACSST5Y2017.S2401.csv'
            )
            S1201_loc = (
                base_loc +
                'ACSST5Y2017.S1201.csv'
            )
            S1101_loc = (
                base_loc +
                'ACSST5Y2017.S1101.csv'
            )
        elif location == "la":
            base_loc = (
                'dataset/census_data/la/2017/'
            )
            B01001_loc = (
                base_loc +
                'ACSDT5YSPT2015.B01001.csv'
            )
            S1401_loc = (
                base_loc +
                'ACSST5Y2017.S1401.csv'
            )
            S2301_loc = (
                base_loc +
                'ACSST5Y2017.S2301.csv'
            )
            S2401_loc = (
                base_loc +
                'ACSST5Y2017.S2401.csv'
            )
            S1201_loc = (
                base_loc +
                'ACSST5Y2017.S1201.csv'
            )
            S1101_loc = (
                base_loc +
                'ACSST5Y2017.S1101.csv'
            )
        elif location == "Chicago, IL":
            base_loc = (
                'dataset/census_data/chicago/2021-acs-5-year-estimate/'
            )
            B01001_loc = (
                base_loc +
                'ACSDT5YSPT2021.B01001-2023-09-13T215358.csv'
            )
            S1401_loc = (
                base_loc +
                'ACSST5Y2021.S1401-2023-09-13T215732.csv'
            )
            S2301_loc = (
                base_loc +
                'ACSST5Y2021.S2301-2023-09-13T215821.csv'
            )
            S2401_loc = (
                base_loc +
                'ACSST5Y2021.S2401-2023-09-13T215905.csv'
            )
            S1201_loc = (
                base_loc +
                'ACSST1Y2021.S1201-2023-09-13T215626.csv'
            )
            S1101_loc = (
                base_loc +
                'ACSST1Y2021.S1101-2023-09-13T230749.csv'
            )
        elif location == "Houston, TX":
            base_loc = (
                'dataset/census_data/houston/2021-acs-5-year-estimate/'
            )
            B01001_loc = (
                base_loc +
                'ACSDT5YSPT2021.B01001-2023-10-14T171425.csv'
            )
            S1401_loc = (
                base_loc +
                'ACSST5Y2021.S1401-2023-10-14T171703.csv'
            )
            S2301_loc = (
                base_loc +
                'ACSST5Y2021.S2301-2023-10-14T171810.csv'
            )
            S2401_loc = (
                base_loc +
                'ACSST5Y2021.S2401-2023-10-14T172115.csv'
            )
            S1201_loc = (
                base_loc +
                'ACSST5Y2021.S1201-2023-10-14T172138.csv'
            )
            S1101_loc = (
                base_loc +
                'ACSST5Y2021.S1101-2023-10-14T172206.csv'
            )
        elif location == "Pittsburgh, PA":
            base_loc = (
                'dataset/census_data/pittsburgh/2021-acs-5-year-estimate/'
            )
            B01001_loc = (
                base_loc +
                'ACSDT5YSPT2021.B01001-2023-10-14T173532.csv'
            )
            S1401_loc = (
                base_loc +
                'ACSST5Y2021.S1401-2023-10-14T174540.csv'
            )
            S2301_loc = (
                base_loc +
                'ACSST5Y2021.S2301-2023-10-14T174611.csv'
            )
            S2401_loc = (
                base_loc +
                'ACSST5Y2021.S2401-2023-10-14T174640.csv'
            )
            S1201_loc = (
                base_loc +
                'ACSST5Y2021.S1201-2023-10-14T174715.csv'
            )
            S1101_loc = (
                base_loc +
                'ACSST5Y2021.S1101-2023-10-14T174733.csv'
            )
        elif location == "Oklahoma City, OK":
            if year == 2021:
                base_loc = (
                    'dataset/census_data/oklahoma_city/'
                    '2021-acs-5-year-estimate/'
                )
                B01001_loc = (
                    base_loc +
                    'ACSDT5YSPT2021.B01001-2024-02-19T000032.csv'
                )
                S1401_loc = (
                    base_loc +
                    'ACSST5Y2021.S1401-2024-02-19T001220.csv'
                )
                S2301_loc = (
                    base_loc +
                    'ACSST5Y2021.S2301-2024-02-19T001543.csv'
                )
                S2401_loc = (
                    base_loc +
                    'ACSST5Y2021.S2401-2024-02-19T001831.csv'
                )
                S1201_loc = (
                    base_loc +
                    'ACSST5Y2021.S1201-2024-02-19T000922.csv'
                )
                S1101_loc = (
                    base_loc +
                    'ACSST5Y2021.S1101-2024-02-19T000808.csv'
                )
            elif year == 2017:
                base_loc = (
                    'dataset/census_data/oklahoma_city/2017/'
                )
                B01001_loc = (
                    base_loc +
                    'ACSDT5YSPT2015.B01001.csv'
                )
                S1401_loc = (
                    base_loc +
                    'ACSST5Y2017.S1401.csv'
                )
                S2301_loc = (
                    base_loc +
                    'ACSST5Y2017.S2301.csv'
                )
                S2401_loc = (
                    base_loc +
                    'ACSST5Y2017.S2401.csv'
                )
                S1201_loc = (
                    base_loc +
                    'ACSST5Y2017.S1201.csv'
                )
                S1101_loc = (
                    base_loc +
                    'ACSST5Y2017.S1101.csv'
                )
            else:
                raise Exception("Invalid year")
        elif location == "Baltimore, MD":
            base_loc = (
                'dataset/census_data/baltimore/2021-acs-5-year-estimate/'
            )
            B01001_loc = (
                base_loc +
                'ACSDT5YSPT2021.B01001-2024-02-25T222226.csv'
            )
            S1401_loc = (
                base_loc +
                'ACSST5Y2021.S1401-2024-02-25T222431.csv'
            )
            S2301_loc = (
                base_loc +
                'ACSST5Y2021.S2301-2024-02-25T222507.csv'
            )
            S2401_loc = (
                base_loc +
                'ACSST5Y2021.S2401-2024-02-25T222540.csv'
            )
            S1201_loc = (
                base_loc +
                'ACSST5Y2021.S1201-2024-02-25T222356.csv'
            )
            S1101_loc = (
                base_loc +
                'ACSST5Y2021.S1101-2024-02-25T222327.csv'
            )
        else:
            raise Exception("Invalid location")

        return (
            B01001_loc, S1401_loc, S2301_loc, S2401_loc, S1201_loc, S1101_loc
        )
