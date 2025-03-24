import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency

def excel_column_letter_to_index(letter):
    letter = letter.upper()
    index = 0
    for char in letter:
        index = index * 26 + (ord(char) - ord('A')) + 1
    return index - 1


def load_data_from_excel(file_path, ages_col1, ages_col2, sex_col1, sex_col2, accept_col1, accept_col2):
    df = pd.read_excel(file_path, skiprows=2)

    ages_col1_idx = excel_column_letter_to_index(ages_col1)
    ages_col2_idx = excel_column_letter_to_index(ages_col2)
    sex_col1_idx = excel_column_letter_to_index(sex_col1)
    sex_col2_idx = excel_column_letter_to_index(sex_col2)
    accept_col1_idx = excel_column_letter_to_index(accept_col1)
    accept_col2_idx = excel_column_letter_to_index(accept_col2)

    ages_cohort1 = df.iloc[:, ages_col1_idx].tolist()
    ages_cohort2 = df.iloc[:, ages_col2_idx].tolist()
    sex_cohort1 = df.iloc[:, sex_col1_idx].tolist()
    sex_cohort2 = df.iloc[:, sex_col2_idx].tolist()
    accept_reject1 = df.iloc[:, accept_col1_idx].tolist()
    accept_reject2 = df.iloc[:, accept_col2_idx].tolist()

    ages_cohort1 = [val for val in ages_cohort1 if pd.notna(val)]
    ages_cohort2 = [val for val in ages_cohort2 if pd.notna(val)]
    sex_cohort1 = [val for val in sex_cohort1 if pd.notna(val)]
    sex_cohort2 = [val for val in sex_cohort2 if pd.notna(val)]
    accept_reject1 = [val for val in accept_reject1 if pd.notna(val)]
    accept_reject2 = [val for val in accept_reject2 if pd.notna(val)]

    return ages_cohort1, ages_cohort2, sex_cohort1, sex_cohort2, accept_reject1, accept_reject2


def filter_data(ages, sex, accept_reject):
    filtered_ages = [age for age, status in zip(ages, accept_reject) if status == 'include']
    filtered_sex = [s for s, status in zip(sex, accept_reject) if status == 'include']
    return filtered_ages, filtered_sex

# Main function to perform the tests
def perform_tests(file_path, ages_col1, ages_col2, sex_col1, sex_col2, accept_col1, accept_col2):
    ages_cohort1, ages_cohort2, sex_cohort1, sex_cohort2, accept_reject1, accept_reject2 = load_data_from_excel(
        file_path, ages_col1, ages_col2, sex_col1, sex_col2, accept_col1, accept_col2
    )

    filtered_ages_cohort1, filtered_sex_cohort1 = filter_data(ages_cohort1, sex_cohort1, accept_reject1)
    filtered_ages_cohort2, filtered_sex_cohort2 = filter_data(ages_cohort2, sex_cohort2, accept_reject2)

    # for agges
    stat, p_value = mannwhitneyu(filtered_ages_cohort1, filtered_ages_cohort2)

    print(f'Mann-Whitney U stat: {stat}')
    print(f'p val: {p_value}')

    alpha = 0.05
    if p_value < alpha:
        print("The two cohorts are not age-matched")
    else:
        print("The two cohorts are age-matched")

    sex_counts_cohort1 = pd.Series(filtered_sex_cohort1).value_counts()
    sex_counts_cohort2 = pd.Series(filtered_sex_cohort2).value_counts()
    contingency_table = pd.DataFrame([sex_counts_cohort1, sex_counts_cohort2], index=['Cohort1', 'Cohort2']).fillna(0)

    # Chi sq for sex match
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    print(f'\nChi-Square stat: {chi2}')
    print(f'p val: {p_value}')

    # Interpretation for sex
    if p_value < alpha:
        print("The two cohorts are not sex-matched.")
    else:
        print("The two cohorts are sex-matched")


file_path = '/Users/markolchanyi/Desktop/DATASET_INFO_20250313.xlsx'

# Column names (or indices) in the Excel file

#### ADNI ####
sex_col1_ADNI = 'AI'
sex_col2_ADNI = 'AO'

ages_col1_ADNI = 'AJ'
ages_col2_ADNI = 'AP'

accept_col1_ADNI = 'AL'
accept_col2_ADNI = 'AR'

#### PPMI ####
sex_col1_PPMI = 'AU'
sex_col2_PPMI = 'BG'

ages_col1_PPMI = 'AV'
ages_col2_PPMI = 'BB'

accept_col1_PPMI = 'AX'
accept_col2_PPMI = 'BD'

#### TBI ####
sex_col1_TBI = 'BM'
sex_col2_TBI = 'BS'

ages_col1_TBI = 'BN'
ages_col2_TBI = 'BT'

accept_col1_TBI = 'BP'
accept_col2_TBI = 'BV'


print("############################### ADNI ###############################")
perform_tests(file_path, ages_col1_ADNI, ages_col2_ADNI, sex_col1_ADNI, sex_col2_ADNI, accept_col1_ADNI, accept_col2_ADNI)
print(" ")
print("############################### PPMI ###############################")
perform_tests(file_path, ages_col1_PPMI, ages_col2_PPMI, sex_col1_PPMI, sex_col2_PPMI, accept_col1_PPMI, accept_col2_PPMI)
print(" ")
print("############################### TBI ###############################")
perform_tests(file_path, ages_col1_TBI, ages_col2_TBI, sex_col1_TBI, sex_col2_TBI, accept_col1_TBI, accept_col2_TBI)
print(" ")
