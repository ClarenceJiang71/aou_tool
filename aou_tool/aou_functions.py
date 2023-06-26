import os
import subprocess
import numpy as np
import pandas as pd

import requests
from collections import defaultdict

# Hail
import hail as hl
# PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# T-SNE
from numpy import reshape
from sklearn.manifold import TSNE

# Used to output correlation visualization
import matplotlib.colors as colors

"""
The first 3 functions are used to interact with Google bucket within the AoU Jupyter Virtual Environment
"""


def copy_from(source_filename: str):
    """
    This is the function to copy a file from google bucket to local virtual environment space
    :param source_filename:
    :return:
    """
    args = ["gsutil", "cp", f"{source_filename}", f"./"]
    output = subprocess.run(args, capture_output=True)

    # print output from gsutil
    output.stderr


def save_cloud_df(df, df_name: str, folder_name: str):
    """
    This is the function to save a pandas dataframe to Google bucket.

    :param df: a dataframe to be saved
    :param df_name: the name of the dataframe file
    :param folder_name: the name of the folder to be stored within the bucket, start after the first /
                        e.g. data/disease/
    """

    my_dataframe = df
    destination_filename = df_name
    my_dataframe.to_csv(destination_filename, index=False)
    my_bucket = os.getenv('WORKSPACE_BUCKET')
    args = ["gsutil", "cp", f"./{destination_filename}", f"{my_bucket}/{folder_name}"]
    output = subprocess.run(args, capture_output=True)
    output.stderr


def save_cloud_file(file_name: str, folder_name: str):
    """
    # This is the function to save any file to Google bucket.

    :param file_name: the file name within the AoU jupyter environment
    :param folder_name: the name of the folder to be stored within the bucket, start after the first /
                        e.g. data/disease/
    """

    destination_filename = file_name
    my_bucket = os.getenv('WORKSPACE_BUCKET')
    args = ["gsutil", "cp", f"./{destination_filename}", f"{my_bucket}/{folder_name}"]
    output = subprocess.run(args, capture_output=True)
    output.stderr


def load_cloud_df(df_name: str, folder_name: str) -> pd.DataFrame:
    """
    # This is the function to load a pandas dataframe from Google bucket

    :param df_name: the name of the dataframe to be loaded
    :param folder_name: the name of the folder that contains the dataframe, start after the first /
                        e.g. data/disease/
    :return: the dataframe
    """
    name_of_file_in_bucket = df_name
    my_bucket = os.getenv('WORKSPACE_BUCKET')
    os.system(f"gsutil cp '{my_bucket}/{folder_name}/{name_of_file_in_bucket}' .")
    print(f'[INFO] {name_of_file_in_bucket} is successfully downloaded into your working space')
    my_dataframe = pd.read_csv(name_of_file_in_bucket)
    return my_dataframe


"""
These 2 functions are used to extract the long table format
"""


def extract_long_table_filtered_version(interest_concept_id: str) -> pd.DataFrame:
    """
    This function is the first way of extracting long table. It only extracts the "person_id", "condition_concept_id",
    and the "standard_concept_name". The main usage is to help calculate the spread format, so it intentionally keeps
    limited information (columns)

    :param interest_concept_id: the interested concept id to be rooted with. The concept_id is provided by the AoU
                                platform, e.g. "4274025"
    :return: the pandas dataframe of the long table.
    """
    dataset_condition_sql_first_part = """
    SELECT 
        c_occurrence.person_id,
        c_occurrence.condition_concept_id,
        c_standard_concept.concept_name as standard_concept_name
    FROM
        ( SELECT
            * 
        FROM
            `""" + os.environ["WORKSPACE_CDR"] + """.condition_occurrence` c_occurrence 
        WHERE
            (
                condition_concept_id IN  (
                    SELECT
                        DISTINCT c.concept_id 
                    FROM
                        `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` c 
                    JOIN
                        (
                            select
                                cast(cr.id as string) as id 
                            FROM
                                `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` cr 
                            WHERE
                                concept_id IN ("""

    # The second part of the AoU searching SQL to retrieve the full long table
    dataset_condition_sql_second_part = """
                                    ) 
                                    AND full_text LIKE '%_rank1]%'
                            ) a 
                                ON (
                                    c.path LIKE CONCAT('%.',
                                a.id,
                                '.%') 
                                OR c.path LIKE CONCAT('%.',
                                a.id) 
                                OR c.path LIKE CONCAT(a.id,
                                '.%') 
                                OR c.path = a.id) 
                            WHERE
                                is_standard = 1 
                                AND is_selectable = 1
                            )
                    )
                ) c_occurrence 
            LEFT JOIN
                `""" + os.environ["WORKSPACE_CDR"] + """.concept` c_standard_concept 
                    ON c_occurrence.condition_concept_id = c_standard_concept.concept_id       
            """

    # This is the auto-generated code by AoU
    dataset_condition_sql = dataset_condition_sql_first_part + interest_concept_id + dataset_condition_sql_second_part
    dataset_condition_df = pd.read_gbq(
        dataset_condition_sql,
        dialect="standard",
        use_bqstorage_api=("BIGQUERY_STORAGE_API_ENABLED" in os.environ),
        progress_bar_type="tqdm_notebook")
    return dataset_condition_df


def extract_long_table_full_version(interest_concept_id: str) -> pd.DataFrame:
    """
    This function is the second way of extracting long table. It extracts every associated concept features,
    including person_id, condition_concept_id, standard_concept_name, standard_concept_code, standard_vocabulary,
    condition_start_datetime, condition_end_datetime, condition_type_concept_id, condition_type_concept_name,
    stop_reason, visit_occurrence_id, visit_occurrence_concept_name, condition_source_value,
    condition_source_concept_id, source_concept_name, source_concept_code, source_vocabulary,
    condition_status_source_value, condition_status_concept_id, condition_status_concept_name.


    :param interest_concept_id: the interested concept id to be rooted with. The concept_id is provided by the AoU
                                platform, e.g. "4274025"
    :return: the pandas dataframe of the long table.
    """
    dataset_condition_sql_first_part = """
    SELECT
        c_occurrence.person_id,
        c_occurrence.condition_concept_id,
        c_standard_concept.concept_name as standard_concept_name,
        c_standard_concept.concept_code as standard_concept_code,
        c_standard_concept.vocabulary_id as standard_vocabulary,
        c_occurrence.condition_start_datetime,
        c_occurrence.condition_end_datetime,
        c_occurrence.condition_type_concept_id,
        c_type.concept_name as condition_type_concept_name,
        c_occurrence.stop_reason,
        c_occurrence.visit_occurrence_id,
        visit.concept_name as visit_occurrence_concept_name,
        c_occurrence.condition_source_value,
        c_occurrence.condition_source_concept_id,
        c_source_concept.concept_name as source_concept_name,
        c_source_concept.concept_code as source_concept_code,
        c_source_concept.vocabulary_id as source_vocabulary,
        c_occurrence.condition_status_source_value,
        c_occurrence.condition_status_concept_id,
        c_status.concept_name as condition_status_concept_name 
    FROM
        ( SELECT
            * 
        FROM
            `""" + os.environ["WORKSPACE_CDR"] + """.condition_occurrence` c_occurrence 
        WHERE
            (
                condition_concept_id IN  (
                    SELECT
                        DISTINCT c.concept_id 
                    FROM
                        `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` c 
                    JOIN
                        (
                            select
                                cast(cr.id as string) as id 
                            FROM
                                `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` cr 
                            WHERE
                                concept_id IN ("""

    # The second part of the AoU searching SQL to retrieve the full long table
    dataset_condition_sql_second_part = """
                                     ) 
                                AND full_text LIKE '%_rank1]%'
                        ) a 
                            ON (
                                c.path LIKE CONCAT('%.',
                            a.id,
                            '.%') 
                            OR c.path LIKE CONCAT('%.',
                            a.id) 
                            OR c.path LIKE CONCAT(a.id,
                            '.%') 
                            OR c.path = a.id) 
                        WHERE
                            is_standard = 1 
                            AND is_selectable = 1
                        )
                )
            ) c_occurrence 
        LEFT JOIN
            `""" + os.environ["WORKSPACE_CDR"] + """.concept` c_standard_concept 
                ON c_occurrence.condition_concept_id = c_standard_concept.concept_id 
        LEFT JOIN
            `""" + os.environ["WORKSPACE_CDR"] + """.concept` c_type 
                ON c_occurrence.condition_type_concept_id = c_type.concept_id 
        LEFT JOIN
            `""" + os.environ["WORKSPACE_CDR"] + """.visit_occurrence` v 
                ON c_occurrence.visit_occurrence_id = v.visit_occurrence_id 
        LEFT JOIN
            `""" + os.environ["WORKSPACE_CDR"] + """.concept` visit 
                ON v.visit_concept_id = visit.concept_id 
        LEFT JOIN
            `""" + os.environ["WORKSPACE_CDR"] + """.concept` c_source_concept 
                ON c_occurrence.condition_source_concept_id = c_source_concept.concept_id 
        LEFT JOIN
            `""" + os.environ["WORKSPACE_CDR"] + """.concept` c_status 
                ON c_occurrence.condition_status_concept_id = c_status.concept_id"""

    # This is the auto-generated code by AoU
    dataset_condition_sql = dataset_condition_sql_first_part + interest_concept_id + dataset_condition_sql_second_part
    dataset_condition_df = pd.read_gbq(
        dataset_condition_sql,
        dialect="standard",
        use_bqstorage_api=("BIGQUERY_STORAGE_API_ENABLED" in os.environ),
        progress_bar_type="tqdm_notebook")
    return dataset_condition_df


"""
These functions are helpful in analyzing the long table. The questions they can answer include: 
1. What are the item counts of all concepts in the long table? 
2. Give a concept_name, what is its concept id? 
"""


def extract_item_count(dataset_condition_df: pd.DataFrame) -> pd.DataFrame:
    """
    This function is used to check a concept's item count (not including the count of its daughter term in the
    hierarchy). E.g. "mental disorder" will have a small item count, although it is a big concept that covers
    anxiety disorder, major depression, etc.

    :param dataset_condition_df: the long table dataframe that shows the concept records of patients
    :return: a panda dataframe that includes 2 columns: the concept name, and the item count.
    """
    dataset_condition_df = dataset_condition_df[["person_id", "condition_concept_id", "standard_concept_name"]]
    dataset_condition_df_drop_duplicates = dataset_condition_df.drop_duplicates()
    item_count_series = dataset_condition_df_drop_duplicates["standard_concept_name"].value_counts()
    item_count_df = item_count_series.reset_index()
    item_count_df.columns = ["Concept Name", "Item Count"]
    return item_count_df


def create_concept_name_and_id_mapping_dictionary(dataset_condition_df: pd.DataFrame, key_column_name: str, value_column_name: str) -> dict:
    """
    This function is used to create a mapping dictionary such that the key is a concept_name and the value is
    the corresponding concept_id, or the other way around, which depends on user preference.
    Within AoU platform, a lot of operations are using concept_id as identifier,
    but human prefer the concept_name, so this function helps set up the connection.

    :param key_column_name: pass either "condition_concept_id" or "standard_concept_name" as the key of mapping dictionary
    :param value_column_name: pass the other column name (from "condition_concept_id" or "standard_concept_name"),
                                different from key_column_name
    :param dataset_condition_df: the dataframe in long table
    :return: a mapping dictionary (key: concept name, value: concept id)
    """
    dataset_condition_df = dataset_condition_df[["person_id", "condition_concept_id", "standard_concept_name"]]
    dataset_condition_df_drop_duplicates = dataset_condition_df.drop_duplicates()
    mapping_dict = dataset_condition_df_drop_duplicates.set_index(key_column_name)[
        value_column_name].to_dict()
    return mapping_dict


"""
These functions are used to extract the spread table dataframe, such that a row represents a participant, 
and each column represents a specific concept, each cell value reflects if a person has a particular concept 
"""


def convert_to_one(x):
    """
    A simple function to convert all values greater than 0 to 1. This function is used to binarize a dataframe.
    It is only used within the "extract_spread_table" function. Do not need to run it separately

    :param x:
    :return:
    """
    if x > 0:
        return 1
    else:
        return x


def extract_spread_table(dataset_condition_df: pd.DataFrame, form: str):
    """
    This function converts a long table dataframe to its spread table form.

    :param dataset_condition_df: the long table dataframe
    :param form: form receives 2 possible string values: "id" or "name". This parameter is only going to change the
                    way how concepts will be represented as columns in the spread table dataframe. If "id" is passed,
                    the columns of the output spread table will show the concept_id of each concept, and if "name" is
                    passed, the columns will show concept_name instead.
    :return: the output is a tuple, where the first value is the spread table dataframe (the cell value reflects
                the frequency of a person getting a concept), and the second value is the binary spread table dataframe
                (the cell value only shows 1 or 0 reflecting if a person has ever had a concept)
    """
    dataset_condition_df = dataset_condition_df[["person_id", "condition_concept_id", "standard_concept_name"]]
    if form == "name":
        grouped_df = dataset_condition_df.groupby(['person_id', 'standard_concept_name']).size().reset_index(
            name='count')
        matrix = grouped_df.pivot_table(index='person_id', columns='standard_concept_name', values="count")
    elif form == "id":
        grouped_df = dataset_condition_df.groupby(['person_id', 'condition_concept_id']).size().reset_index(
            name='count')
        matrix = grouped_df.pivot_table(index='person_id', columns='condition_concept_id', values="count")
    else:
        print("please enter name or id as parameter values for form")
        return
    matrix = matrix.fillna(0)
    binary_matrix = matrix.applymap(convert_to_one)
    return matrix, binary_matrix


"""
These functions are used to calculate the roll-up count of a concept based on a spread table dataframe.
The first 3 functions are helper methods to make the "extract_roll_up_count" function work.
"""


def get_proper_ancestors(cid, concept_ancestor_df):
    """
    This is just a function to help calculate roll-up count.
    """
    filtered_df = concept_ancestor_df[concept_ancestor_df['descendant_concept_id'] == cid]
    filtered_df = filtered_df[filtered_df['ancestor_concept_id'] != cid]
    ancestors_list = filtered_df['ancestor_concept_id'].tolist()
    return ancestors_list


def get_descendants(cid, concept_ancestor_df):
    """
    This is just a function to help calculate roll-up count
    """
    filtered_df = concept_ancestor_df[concept_ancestor_df['ancestor_concept_id'] == cid]
    descendants_list = filtered_df['descendant_concept_id'].tolist()
    return descendants_list


def set_context(root_cids, binary_spread_table, concept_ancestor_df):
    """
    This is just a function to help calculate roll-up count.

    :param root_cids: a list of rooted concept ids
    :param binary_spread_table: the spread table dataframe in binary form
    :param concept_ancestor_df: a given dataframe that stores the ancestor relationship between concepts,
                                this dataframe is imported as a csv file, you have to read in this csv file within
                                your environment before running this function!
    :return: a cleaned-up version of the concept ancestor dataframe
    """
    # first, remove the cids that are not descendants of any root
    visible = set()

    for cid in root_cids:

        new_visible = get_descendants(cid, concept_ancestor_df)
        #         print(f"new visible:")
        #         print(new_visible)
        new_visible.append(cid)

        for item in new_visible:
            visible.add(item)

    new_concept_ancestor_df = concept_ancestor_df[concept_ancestor_df['descendant_concept_id'].isin(visible)]
    new_concept_ancestor_df = new_concept_ancestor_df[new_concept_ancestor_df['ancestor_concept_id'].isin(visible)]

    # then, remove the cids that are not columns in the item_count dataframe
    new_concept_ancestor_df = new_concept_ancestor_df[
        new_concept_ancestor_df['descendant_concept_id'].isin(binary_spread_table.columns)]
    new_concept_ancestor_df = new_concept_ancestor_df[
        new_concept_ancestor_df['ancestor_concept_id'].isin(binary_spread_table.columns)]

    return new_concept_ancestor_df


def extract_roll_up_count(root_cids: list, binary_spread_table: pd.DataFrame, concept_ancestor_df: pd.DataFrame) \
        -> pd.DataFrame:
    """
    This is a function to calculate roll up count (adding up all the item counts of the concepts within the
    hierarchy rooted at a selective concept). For example, the item count of "mental disorder" is only about 2k,
    but the rollup count is about millions since it adds up sub-concepts like anxiety disorder, major depression, etc.
    The roll-up value needs to be shown with the print

    :param root_cids: a list of rooted concept ids
    :param binary_spread_table: the spread table dataframe in binary form
    :param concept_ancestor_df: a given dataframe that stores the ancestor relationship between concepts,
                                this dataframe is imported as a csv file, you have to read in this csv file within
                                your environment before running this function!
    :return: a rollup count dataframe, row represents a patient, column represents a concept, a cell at (row, column)
            will have 1 if this person has either this concept or a concept that is under this concept's hierarchy.
    """
    clean_concept_ancestor_df = set_context(root_cids, binary_spread_table, concept_ancestor_df)
    rolled_up_df = binary_spread_table.copy()
    # for each cid column, we add its item_count to each ancestor item_count
    for cid in binary_spread_table.columns:
        ancestors = get_proper_ancestors(cid, clean_concept_ancestor_df)
        for ancestor in ancestors:
            rolled_up_df[ancestor] = rolled_up_df[ancestor] | binary_spread_table[cid]
    return rolled_up_df


def print_roll_up(roll_up_df: pd.DataFrame, mapping_dict: dict):
    """
    # This is the code to collectively print out results of rollup counts based on the roll up dataframe
    # outputted from the "extract_roll_up_count" functions

    :param mapping_dict: mapping_dict is required to be in the form of "concept id-concept name" key pair, because all
                        functions related to roll-up calculation are connected through concept id instead of concept
                        name, but at the final presentation, it is hard to verify your result without knowing the
                        corresponded name of the concept id, so mapping_dict is required in this form.

                        mapping_dict could be obtained through the "create_concept_name_and_id_mapping_dictionary"
                        function
    :param roll_up_df: the roll_up_df obtained through the extract_roll_up_count function
    """
    col_sums = roll_up_df.sum().sort_values(ascending=False)
    sorted_roll_up_df = roll_up_df[col_sums.index]

    for col in sorted_roll_up_df.columns:
        col_sum = roll_up_df[col].sum()
        print(f"{mapping_dict[col]}: {col_sum}")


"""
These functions are used to help EDA
1. Extract Summary Count 
2. Correlation 
3. PCA
4. t-SNE
"""


def get_filter_df_by_item_count(df: pd.DataFrame, item_count_df: pd.DataFrame, threshold: int)-> pd.DataFrame:
    """
    # This is the function that filter the columns (concepts) of a spread table based on the concept's item count

    :param df: the spread table dataframe
    :param item_count_df: the dataframe that stores the item count of each concept, obtained by
                            the function "extract_item_count"
    :param threshold: the threshold value of item count to filter with
    :return: a filtered form of the spread table, essentially, some columns are filtered.
    """
    item_count_filtered_series = item_count_df[item_count_df["Item Count"] >= threshold]
    columns = item_count_filtered_series["Concept Name"]
    filter_condition = df.columns.isin(columns)
    spread_table_filtered = df.loc[:, filter_condition]
    return spread_table_filtered


def overlap_coefficient(x, y):
    """
    This is the function to be used within the "correlation_visualization_generation" function
    when the metric parameter is set to "overlap". It replaces the default "pearson correlation" with
    "overlap coefficient".

    Should not use this function separately.
    """
    intersection = np.logical_and(x, y).sum()
    min_value = min(x.sum(), y.sum())

    if min_value != 0:
        overlap = intersection / min_value
    else:
        overlap = 0

    return overlap


def jaccard_similarity(x, y):
    """
     This is the function to be used within the "correlation_visualization_generation" function
     when the metric parameter is set to "jaccard". It replaces the default "pearson correlation" with
     "jaccard coefficient".

     Should not use this function separately.
     """
    intersection = np.logical_and(x, y).sum()
    union = np.logical_or(x, y).sum()

    if union != 0:
        jaccard = intersection / union
    else:
        jaccard = 0

    return jaccard


def correlation_visualization_generation_filter_by_1000(metric: str, spread_table_binary: pd.DataFrame, item_count_all: pd.DataFrame, concept_name: str):
    """

    This visualization function aims to use the binary spread table dataframe to create
    correlation visualizations, but generally a dataframe will just have too many concepts,
    so due to this large amount of column, visualization is hard. Thus, this function also
    further filters the concept to keep those with item count > 1000, as a way
    to reduce the number of concepts involved in visualization.

    3 types of visualization will be created, a correlation heatmap, a clustermap of the correlation matrix
    to re-order the concepts, and a pair count to show the number of participants who have a
    pair of concepts

    The image files will be stored locally in the Jupyter environment, if you want to store
    it in Google bucket, you could do a separate store "save_cloud_file"

    :param concept_name: the concept name of the major root of the whole hierarchy of the spread table dataframe,
                        e.g. mental disorder
    :param spread_table_binary: the binary spread table dataframe
    :param item_count_all: the dataframe that stores the item count of each concept
    :param metric: a str with 3 possible values: "pearson", "overlap" and "jaccard" to indicate
                    which type of correlation metric is going to be used

    """
    # Get the binary spread table that only keeps the concepts that have item count >= 1000
    spread_table_binary_filter_1000 = get_filter_df_by_item_count(spread_table_binary, item_count_all, 1000)
    spread_table_binary_filter_1000 = spread_table_binary_filter_1000.astype(int)

    if metric == "pearson":
        corr_matrix = spread_table_binary_filter_1000.corr()
    elif metric == "overlap":
        corr_matrix = spread_table_binary_filter_1000.corr(method=overlap_coefficient)
    elif metric == "jaccard":
        corr_matrix = spread_table_binary_filter_1000.corr(method=jaccard_similarity)

    # heatmap creation
    # 1.set mask to filter out values on or above diagonal
    mask_hm = np.tri(corr_matrix.shape[0], k=-1).T
    np.fill_diagonal(mask_hm, 1)

    # 2. set color
    cmap = colors.LinearSegmentedColormap.from_list('custom_cmap', ['white', 'orange', 'darkred'], N=256)

    # 3. create heatmap
    plt.figure(figsize=(32, 24))
    ax = sns.heatmap(corr_matrix, mask=mask_hm, cmap=cmap, linewidth=0.5)
    plt.title(f'{concept_name} {metric} Correlation Matrix')

    # 4. save file and show plot
    hm_file_name = f"{concept_name}_{metric}_heatmap_binary_spread_table_filter_more_than_1000.png"
    plt.savefig(hm_file_name)
    plt.show()

    # 5. clustermap creation
    # 1. set mask to filter out values on diagonal
    mask_cm = np.eye(*corr_matrix.shape, dtype=bool)
    # 2. create clustermap
    ax = sns.clustermap(corr_matrix, mask=mask_cm, cmap=cmap, figsize=(40, 30))
    # 3. save file and show plot
    plt.title(f'{concept_name} {metric} Cluster Map')
    cm_file_name = f"{concept_name}_{metric}_clustermap_binary_spread_table_filter_more_than_1000.png"
    plt.savefig(cm_file_name)
    plt.show()

    if metric == "jaccard":
        # 6. create the heatmap that shows the count of patients having both diseases
        output_df = pd.DataFrame(index=spread_table_binary_filter_1000.columns,
                                 columns=spread_table_binary_filter_1000.columns, dtype=int)
        for col1 in spread_table_binary_filter_1000.columns:
            for col2 in spread_table_binary_filter_1000.columns:
                count = np.logical_and(spread_table_binary_filter_1000[col1],
                                       spread_table_binary_filter_1000[col2]).sum()
                output_df.loc[col1, col2] = count

        output_df = output_df.astype(int)
        mask = np.tri(output_df.shape[0], k=-1).T
        cmap = colors.LinearSegmentedColormap.from_list(
            'custom_cmap', ['white', 'orange', 'darkred'], N=256)
        np.fill_diagonal(mask, 1)
        plt.figure(figsize=(64, 48))
        ax = sns.heatmap(output_df, mask=mask, cmap=cmap, linewidth=1, annot=True, fmt='.0f',
                         annot_kws={'size': 12})
        plt.title(f'{concept_name} Pair Count Matrix')
        plt.savefig(f"{concept_name}_pair_count_heatmap_binary_spread_table_filter_more_than_1000.png")
        plt.show()

"""
These three funtions help extract disease tables and combine them into a large hail table that could efficiently stores and processes data
"""
def process_disease_concept_id_list(disease_concept_id_list, disease_list, version, folder_name):
    """
    Process a list of disease concept IDs and associated disease names to extract long tables and save them in a specific folder in google cloud bucket.

    Args:
        disease_concept_id_list (list): A list of disease concept IDs.
        disease_list (list): A list of corresponding disease names.
        version (str): The version number for the AoU dataset.
        folder_name (str): The folder path in google cloud bucket where the long tables will be saved.

    Returns:
        None

    """
    for i in range(len(disease_concept_id_list)):
        long_table = extract_long_table_full_version(disease_concept_id_list[i])
        concept_name = disease_list[i]
        name = f"2023-06-15_aou-v{version}_{concept_name}_long-table.tsv"
        save_cloud_df(long_table, name, folder_name)


def load_cloud_ht(table_name, folder_name):
    """
    Load a Hail Table (HT) from a cloud storage bucket and import it into Hail.

    Args:
        table_name (str): The name of the table to be loaded.
        folder_name (str): The folder path in the cloud storage bucket where the table is located.

    Returns:
        my_ht (hail.Table): The Hail Table loaded from the cloud storage bucket.

    """
    name_of_file = table_name
    my_bucket = os.getenv('WORKSPACE_BUCKET')
    os.system(f"gsutil cp '{my_bucket}/{folder_name}/{name_of_file}' .")
    my_ht = hl.import_table(name_of_file, impute = False, delimiter = '\t')
    return my_ht


def load_tables(disease_list, version):
    """
    Load tables from a list of diseases.

    Parameters:
        disease_list (list): A list of diseases.
        version (float): The version number of the tables to be loaded.

    Returns:
        table_list: A list containing all disease hail tables.
    """
    filename_list = []
    for i in range(len(disease_list)):
        concept_name = disease_list[i]
        file_name = f"2023-06-15_aou-v{version}_{concept_name}_long-table.tsv"
        filename_list.append(file_name)

    folder_name = 'data/exports/2023-06-15_disease/tsv'
    table_list = []
    for i in range(len(filename_list)):
        file_name = filename_list[i]
        ht = load_cloud_ht(file_name, folder_name)
        table_list.append(ht)
        
    return table_list

def get_long_table(table_list):
    """
    Combines multiple Hail tables into a single long table.

    This function takes a list of Hail tables and combines them into a single long table by performing a union operation.
    The resulting long table will contain all the rows from each input table.

    Args:
        table_list (list): A list of Hail tables to be combined into a long table.

    Returns:
        hl.Table: The combined long table.

    Example:
        # Assuming you have three Hail tables: table1, table2, and table3
        combined_table = get_long_table([table1, table2, table3])
        # combined_table will be the long table containing rows from table1, table2, and table3.
    """
    disease_table = table_list[0]
    for table in table_list[1:]:
        disease_table = disease_table.union(table)
    
    return disease_table


def add_source_concept_column(table_list, disease_list):
    """
    Adds a source_concept column to each Hail table in the input list.

    This function takes a list of Hail tables and a corresponding list of source concepts and adds a new column called
    'source_concept' to each table. The 'source_concept' column is populated with the respective source concept value
    from the disease_list. The updated tables, each containing the new 'source_concept' column, are returned as a list.

    Args:
        table_list (list): A list of Hail tables to which the source_concept column will be added.
        disease_list (list): A list of source concepts, where each element corresponds to the respective Hail table in
                             table_list.

    Returns:
        list: A list of Hail tables with the added 'source_concept' column.

    Example:
        # Assuming you have two Hail tables: table1 and table2, and a corresponding list of source concepts: diseases
        updated_tables = add_source_concept([table1, table2], diseases)
        # updated_tables will be a list containing table1 and table2 with the 'source_concept' column added.
    """
    table_list_withsc = []
    for i in range(len(table_list)):
        source_concept_cur = disease_list[i]
        table_withsc = table_list[i].annotate(source_concept=source_concept_cur)
        table_list_withsc.append(table_withsc)
    
    return table_list_withsc
