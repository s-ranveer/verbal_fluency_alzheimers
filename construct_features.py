# This is the file for constructing the features for the Alzheimer's speech dataset
import json
import os
import re
import pandas as pd
import wordfreq
import nltk
import pronouncing
from sklearn.cluster import KMeans
from tqdm import tqdm


input_dir = "/home/rxs174730/programming/speech/outputs/transcriptions_wo_speakers/year_1"
aoa_path = "data/age_of_acquisition.xlsx"
aoa_sec_path = "data/age_of_acquisition_secondary.xlsx"
output_path = "/home/rxs174730/programming/speech/outputs/features_year_1.csv"
binned_output_path = "/home/rxs174730/programming/speech/outputs/features_year_1_binned.csv"

def word_frequency(response: list, aggregate: str="mean", letter=None, semantic_category=None, **kwargs):
    # We would use the wordfreq library to calculate the frequency of words in the response
    """
    Calculate the average word frequency in a response.
    :param response: The processed response string
    :param aggregate: Specify "mean" for average word frequency or "total" for total frequency of all words
    :param letter: If the letter is provided, only use words which begin with it
    :return: Mean word frequency or total frequency of all words as wella s the total number of words considered for the calculation
    """
    words = [r.strip().lower() for r in response]
    if letter is not None:
        words = [w for w in words if w.startswith(letter)]
    if semantic_category is not None and semantic_category in kwargs:
        category_words = kwargs[semantic_category]
        words = [w for w in words if w in category_words]
    words = list(set(words))  # Consider unique words only
    frequencies = [wordfreq.word_frequency(word, "en") for word in words]
    if aggregate == "total":
        return sum(frequencies), len(frequencies)
    elif aggregate == "mean":
        return sum(frequencies) / len(frequencies) if len(frequencies) > 0 else 0.0, len(frequencies)
    else:
        raise NotImplementedError(f"Aggregate method {aggregate} not implemented. Use 'mean' or 'total'.")

def word_length(response: list, aggregate: str="mean", letter=None, semantic_category=None, **kwargs):
    """Calculate the length of an average word in a response.
    :param response: The processed response string
    :param aggregate: Specify "mean" for average word length or "total" for total length of all words
    :param letter: If the letter is provided, only use words which begin with it
    :return: Mean word length or total length of all words as well as the total number of words considered for the calculation
    """
    words = [r.strip().lower() for r in response]
    if letter is not None:
        words = [w for w in words if w.startswith(letter)]
    if semantic_category is not None and semantic_category in kwargs:
        category_words = kwargs[semantic_category]
        words = [w for w in words if w in category_words]
    words = list(set(words))  # Consider unique words only
    lengths = [len(word) for word in words]
    if aggregate == "mean":
        return sum(lengths) / len(lengths) if lengths else 0.0, len(lengths)
    elif aggregate == "total":
        return sum(lengths), len(lengths)
    else:
        raise NotImplementedError(f"Aggregate method {aggregate} not implemented. Use 'mean' or 'total'.")

def age_of_acquisition_secondary(word, aoa_sec_df):
    # We would try to get the row where the word or its lemma match the "WORD" column in the secondary AoA data file and get the "AoA" value from that row
    row = aoa_sec_df[aoa_sec_df["WORD"].str.lower() == word.lower()]
    if row.empty:
        # Try lemmatized version of the word
        lemmatizer = nltk.WordNetLemmatizer()
        lemma = lemmatizer.lemmatize(word)
        row = aoa_sec_df[aoa_sec_df["WORD"].str.lower() == lemma.lower()]
        if row.empty:
            return None
        else:
            AOA_values = row["AoAtestbased"].values
            if len(AOA_values) > 0:
                # Return the smallest AoA value if there are multiple entries for the same word
                return min(AOA_values) if not pd.isnull(AOA_values).all() else None
            else:
                return None
    else:
        AOA_values = row["AoAtestbased"].values
        if len(AOA_values) > 0:
            # Return the smallest AoA value if there are multiple entries for the same word
            return min(AOA_values) if not pd.isnull(AOA_values).all() else None
        else:
            return None

def age_of_acquisition(response: list, aoa_path: str, aoa_sec_path, aggregate: str="mean", letter = None, semantic_category=None, **kwargs):
    """Calculate the average age of acquisition of words in a response.
    :param response: The processed response string
    :param aoa_path: Path to the age of acquisition data file
    :param aoa_sec_path: Path to the secondary age of acquisition data file to be used if the word is not found in the primary file
    :param aggregate: Specify "mean" for average AoA or "total" for total AoA of all words
    :param letter: If the letter is provided, only use words which begin with it
    :return: Mean AoA or total AoA of all words as well as the total number of words considered for the calculation
    """
    words = [r.strip().lower() for r in response]
    words = [w.replace("_", " ") for w in words]  # Replace underscores with spaces if present
    if letter is not None:
        words = [w for w in words if w.startswith(letter)]
    if semantic_category is not None and semantic_category in kwargs:
        category_words = kwargs[semantic_category]
        words = [w for w in words if w in category_words]
    words = list(set(words))  # Consider unique words only
    # Use nltk to lemmatize words
    lemmatizer = nltk.WordNetLemmatizer()
    lemmas = {word: lemmatizer.lemmatize(word) for word in words}
    # Load age of acquisition data which is an xlsx file (Sheet1)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Cannot parse header or footer")
        aoa_data = pd.read_excel(aoa_path, sheet_name="Sheet1", header=0, skipfooter=0)
        aoa_sec_df = pd.read_excel(aoa_sec_path, sheet_name="a", header=0, skipfooter=0)
    # We are concerned with the columns "Word", "Alternative.spelling", "Lemma_highest_PoS", "AoA_Kup", and "AoA_Kup_lem"
    word_aoa = {}
    for word, lemma in lemmas.items():
        # We would first try to find the AoA using the original word
        row = aoa_data[aoa_data["Word"].str.lower() == word.lower()]
        if row.empty:
            # If not found, try alternative spelling
            row = aoa_data[aoa_data["Alternative.spelling"].str.lower() == word.lower()]
            if row.empty:
                # If still not found, try lemma
                # We would use the "Lemma_highest_PoS" column for this
                row = aoa_data[aoa_data["Lemma_highest_PoS"].str.lower() == lemma.lower()]
                if row.empty:
                    # We would assign the AoA as None if not found
                    word_aoa[word] = age_of_acquisition_secondary(word, aoa_sec_df)
                else:
                    if not row["AoA_Kup_lem"].isnull().values[0]:
                        word_aoa[word] = row["AoA_Kup_lem"].values[0]
                    else:
                        word_aoa[word] = age_of_acquisition_secondary(word, aoa_sec_df)
            else:
                if not row["AoA_Kup"].isnull().values[0]:
                    word_aoa[word] = row["AoA_Kup"].values[0]
                elif not row["AoA_Kup_lem"].isnull().values[0]:
                    word_aoa[word] = row["AoA_Kup_lem"].values[0]
                else:
                    word_aoa[word] = age_of_acquisition_secondary(word, aoa_sec_df)
        else:
            if not row["AoA_Kup"].isnull().values[0]:
                word_aoa[word] = row["AoA_Kup"].values[0]
            elif not row["AoA_Kup_lem"].isnull().values[0]:
                word_aoa[word] = row["AoA_Kup_lem"].values[0]
            else:
                word_aoa[word] = age_of_acquisition_secondary(word, aoa_sec_df)
    # Now calculate the aggregate AoA
    aoa_values = [aoa for aoa in word_aoa.values() if aoa is not None]
    if not aoa_values:
        return 0.0, 0
    if aggregate == "mean":
        return float(sum(aoa_values) / len(aoa_values)), len(aoa_values)
    elif aggregate == "total":
        return float(sum(aoa_values)), len(aoa_values)
    else:
        raise NotImplementedError(f"Aggregate method {aggregate} not implemented. Use 'mean' or 'total'.")

def neigborhood_density(response: list, clustering_type: str="semantic", letter=None, semantic_category=None, **kwargs) -> dict:
    """
    Docstring for neigborhood_density
    
    :param response: The response to the question
    :type response: list
    :param clustering_type: The type of clustering, Phonetic and Semantic to perform
    :type clustering_type: str
    :param letter: The letter to filter the words list if provided
    :param kwargs: Additional arguments
    :return: Return the number of switches, average cluster size and total number of words
    :rtype: dict[Any, Any]
    """
    wordnet_lemmatizer = nltk.WordNetLemmatizer()
    words = [r.strip().lower() for r in response]
    words = [w.replace("_", " ") for w in words]  # Replace underscores with spaces if present
    if letter is not None:
        words = [w for w in words if w.startswith(letter)]
    if semantic_category is not None and semantic_category in kwargs:
        words = [wordnet_lemmatizer.lemmatize(w) for w in words]  # Lemmatize words for better matching with category words
        category_words = kwargs[semantic_category]
        words = [w for w in words if w in category_words]
    if clustering_type == "semantic":
        # We would consider the list of words generated from the response and create clusters on consecutive words based on
        # whether they 
        assert semantic_category in ["animal", "vegetable"], "Semantic category should be either 'animal' or 'vegetable'"
        if kwargs and f"{semantic_category}_groups" in kwargs:
            # Once we have the current cluster
            group_dict = kwargs[f"{semantic_category}_groups"]
            # We would assign each word to all the possible groups it belongs to 
            word_groups = {}
            for word in words:
                word_groups[word] = []
                for group_id, group_words in group_dict.items():
                    if word in group_words:
                        word_groups[word].append(group_id)
            
    elif clustering_type == "phonetic":
        # If the words are to clustered based on phonetic similarity, we would do the following 
        # If the two words have the same first two letters, or differ by a vowel sound, rhyme or are homonyms, 
        # they belong to the same cluster
        
        # Assign groups based on first two letters
        group_dict = {}
        for word in words:
            first_two = f"FT_{word[:2]}"
            if first_two not in group_dict:
                group_dict[first_two] = []
            group_dict[first_two].append(word)

        # Assign groups based on pronounciation feature
        for w1 in words:
            for w2 in words:
                if w1 == w2:
                    continue
                phones_w1 = pronouncing.phones_for_word(w1)
                phones_w2 = pronouncing.phones_for_word(w2)

                # Check for vowel sound difference, rhyme, or homonymy
                if (pronouncing.rhymes(w1) and w2 in pronouncing.rhymes(w1)):
                    group_id = f"RHYME_{w1}"
                    if group_id not in group_dict:
                        group_dict[group_id] = [w1]
                    group_dict[group_id].append(w2)
                
                # Check if the words are homonyms
                elif phones_w1 and set(phones_w1) == set(phones_w2):
                    group_id = f"HOMONYM_{str(set(phones_w1))}"
                    if group_id not in group_dict:
                        group_dict[group_id] = [w1]
                    group_dict[group_id].append(w2)
                
                else:
                    # Check for the vowel sound difference. If the words differ by only one vowel sound and everything else is the same
                    # we would consider them to be in the same group with everything other than the vowel sound making the group id
                    if phones_w1 and phones_w2:
                        # Find differences in phonetic representation across different pronunciations
                        for pron1 in phones_w1:
                            for pron2 in phones_w2:
                                if len(pron1) != len(pron2):
                                    continue
                                else:
                                    diff_count = 0
                                    diff_positions = []
                                    for i, (p1, p2) in enumerate(zip(pron1.split(), pron2.split())):
                                        if p1 != p2:
                                            diff_count += 1
                                            diff_positions.append(i)
                                    if diff_count == 1:
                                        is_vowel_diff = False
                                        diff_index = diff_positions[0]

                                        # The vowels in ARPAbet have numbers in them
                                        if pron1.split()[diff_index][-1].isdigit() and pron2.split()[diff_index][-1].isdigit():
                                            is_vowel_diff = True
                                        if is_vowel_diff:
                                            # The group id would be based on everything other than the differing vowel sound
                                            group_id = f"VOWEL_DIFF_{w1}"
                                            if group_id not in group_dict:
                                                group_dict[group_id] = [w1]
                                            group_dict[group_id].append(w2)
                                            break
                    else:
                        continue
        
        # We would assign each word to all the possible groups it belongs to 
        word_groups = {}
        for word in words:
            word_groups[word] = []
            for group_id, group_words in group_dict.items():
                if word in group_words:
                    word_groups[word].append(group_id)
    else:
        raise NotImplementedError(f"Clustering type {clustering_type} not implemented. Use 'semantic' or 'phonetic'.")
    
    # Now we would assign clusters based on consecutive words,
    # If two consecutive words share at least one group, they belong to the same cluster
    clusters = {}
    current_cluster_id = 0
    for i in range(len(words)):
        word = words[i]
        # First word always starts a new cluster
        if i == 0:
            clusters[current_cluster_id] = [word]

        # If the current word shares groups with the previous word, they belong to the same cluster
        elif set(word_groups[words[i]]).intersection(set(word_groups[words[i-1]])):
            # We would check if the current word shares groups with all the words in the current cluster
            # Otherwise, the previous word would belong to an additional cluster with the current word
            create_new_cluster = False
            for word_in_cluster in sorted(clusters[current_cluster_id], reverse=True):
                # If the current cluster has a word that does not share any group with the current word
                # we would need to create a new cluster
                if not set(word_groups[word_in_cluster]).intersection(set(word_groups[word])):
                    create_new_cluster = True
                    break

            # Depending on whether we need to create a new cluster or not when the current word doesn't share groups with all words in the current cluster
            if create_new_cluster:
                current_cluster_id += 1
                clusters[current_cluster_id] = [word]
                # If we need to create a new cluster, we would need to determine which words would also belong to this new cluster
                for word_in_cluster in sorted(clusters[current_cluster_id-1], reverse=True):
                    if set(word_groups[word_in_cluster]).intersection(set(word_groups[word])):
                        clusters[current_cluster_id].insert(0, word_in_cluster)  # Add the word to the new cluster in the front
                    else:
                        break
            else:
                clusters[current_cluster_id].append(word)

        # If the current word does not share any groups with the previous word, it starts a new cluster
        else:
            current_cluster_id += 1
            clusters[current_cluster_id] = [word]
    
    num_switches = current_cluster_id
    avg_cluster_size = sum(len(cluster)-1 for cluster in clusters.values()) / len(clusters) if clusters else 0.0
    return {"num_switches": num_switches, "avg_cluster_size": avg_cluster_size, "total_words": len(set(words))}

def pause_rate(pauses, pause_threshold_in_seconds: float, aggregate: str="mean"):
    """
    Calculate pause features from the list of pauses.
    :param pauses: List of pauses with their start and end times.
    :param pause_threshold_in_seconds: Threshold to consider a pause significant. Pauses lower than the thtshold are ignored.
    :param aggregate: Description of the aggregate method to use ("mean" or "total")
    :return: The calculated pause feature based on the specified aggregate method as well as the total number of significant pauses considered for the calculation
    """
    # The pauses are a list of dictionaries with pause start and end times
    for pause in pauses:
        pause_duration = float(pause["end"]) - float(pause["start"])
        pause["duration"] = pause_duration
    significant_pauses = [pause["duration"] for pause in pauses if pause["duration"] >= pause_threshold_in_seconds]
    if not significant_pauses:
        return 0.0, 0
    if aggregate == "mean":
        return sum(significant_pauses) / len(significant_pauses), len(significant_pauses)
    elif aggregate == "total":
        return sum(significant_pauses), len(significant_pauses)
    else:
        raise NotImplementedError(f"Aggregate method {aggregate} not implemented. Use 'mean' or 'total'.")

def speech_rate(raw_response, time_segments):
    """
    Calculate speech rate as words per second.
    :param raw_response: The raw response text from which we can calculate the total number of words spoken
    :param time_segments: The list of time segments corresponding to the responses
    :return: The speech rate in words per second as well as the total number of words considered for the calculation
    """
    words = nltk.word_tokenize(raw_response)
    words = [word for word in words if word.isalnum()]  # Consider only alphanumeric tokens as words
    total_words = len(words)

    total_time = sum(float(segment["end"]) - float(segment["start"]) for segment in time_segments)
    if total_time == 0:
        return 0.0, 0
    return total_words / total_time, total_time

def process_data(response_data: dict, aoa_path: str, aoa_sec_path: str, clustering_data: dict) -> dict:
    """
    Process the response data to extract features.
    :param response_data: The processed response data for a patient
    :param aoa_path: Path to the age of acquisition data file
    :param aoa_sec_path: Path to the secondary age of acquisition data file to be used if the word is not found in the primary file
    :param clustering_data: The data required for clustering (e.g., animal and vegetable groups)
    :return: A dictionary containing the extracted features
    """
    # The dictionary has keys, R1, R2, r3 and 4r4 corresponding to letter f ,letter L, Animals and vegetables respectively
    features = {}

    # Process for R1 and R2 which are letter f and letter L respectively
    for response_key in ["R1", "R2"]:
        if response_key in response_data and response_data[response_key]:
            if response_key == "R1":
                letter = "f"
            else:
                letter = "l"
            features[f"{response_key}_word_frequency_mean"], features[f"{response_key}_word_frequency_total_words"] = word_frequency(response_data[response_key]["extracted_answer"], aggregate="mean", letter=letter)
            features[f"{response_key}_word_length_mean"], features[f"{response_key}_word_length_total_words"] = word_length(response_data[response_key]["extracted_answer"], aggregate="mean", letter=letter)
            features[f"{response_key}_age_of_acquisition_mean"], features[f"{response_key}_age_of_acquisition_total_words"] = age_of_acquisition(response_data[response_key]["extracted_answer"], aoa_path, aoa_sec_path=aoa_sec_path, aggregate="mean", letter=letter)
            cluster_metrics = neigborhood_density(response_data[response_key]["extracted_answer"], clustering_type="phonetic", letter=letter)
            features[f"{response_key}_num_switches"] = cluster_metrics["num_switches"]
            features[f"{response_key}_avg_cluster_size"] = cluster_metrics["avg_cluster_size"]
            features[f"{response_key}_total_words"] = cluster_metrics["total_words"]
            features[f"{response_key}_pause_rate"], features[f"{response_key}_pause_rate_total_pauses"] = pause_rate(response_data[response_key]["pauses"], pause_threshold_in_seconds=0.5, aggregate="mean")
            features[f"{response_key}_speech_rate"], features[f"{response_key}_speech_rate_total_time"] = speech_rate(response_data[response_key]["full_response"], response_data[response_key]["response_timestamps"])
        
    for response_key in ["R3", "R4"]:
        if response_key == "R3":
            semantic_category = "animal"
        else:
            semantic_category = "vegetable"
        if response_key in response_data and response_data[response_key]:
            features[f"{response_key}_word_frequency_mean"], features[f"{response_key}_word_frequency_total_words"] = word_frequency(response_data[response_key]["extracted_answer"], aggregate="mean", semantic_category=semantic_category, **clustering_data)
            features[f"{response_key}_word_length_mean"], features[f"{response_key}_word_length_total_words"] = word_length(response_data[response_key]["extracted_answer"], aggregate="mean", semantic_category=semantic_category, **clustering_data)
            features[f"{response_key}_age_of_acquisition_mean"], features[f"{response_key}_age_of_acquisition_total_words"] = age_of_acquisition(response_data[response_key]["extracted_answer"], aoa_path, aoa_sec_path=aoa_sec_path, aggregate="mean", semantic_category=semantic_category, **clustering_data)
            cluster_metrics = neigborhood_density(response_data[response_key]["extracted_answer"], clustering_type="semantic", semantic_category = semantic_category, **clustering_data)
            features[f"{response_key}_num_switches"] = cluster_metrics["num_switches"]
            features[f"{response_key}_avg_cluster_size"] = cluster_metrics["avg_cluster_size"]
            features[f"{response_key}_total_words"] = cluster_metrics["total_words"]
            features[f"{response_key}_pause_rate"], features[f"{response_key}_pause_rate_total_pauses"] = pause_rate(response_data[response_key]["pauses"], pause_threshold_in_seconds=0.5, aggregate="mean")
            features[f"{response_key}_speech_rate"], features[f"{response_key}_speech_rate_total_time"] = speech_rate(response_data[response_key]["full_response"], response_data[response_key]["response_timestamps"])
    
    for feature_type, q1, q2 in [("phonetic", "R1", "R2"), ("semantic", "R3", "R4")]:
        # Get the overall number of switches
        features[f"{feature_type}_num_switches"] = features.get(f"{q1}_num_switches", 0) + features.get(f"{q2}_num_switches", 0)

        # Get the overall phonetic or semantic  specific average cluster size.
        if all(key in features for key in [f"{q1}_avg_cluster_size", f"{q2}_avg_cluster_size"]):
            features[f"{feature_type}_avg_cluster_size"] = (features[f"{q1}_avg_cluster_size"]*(features[f"{q1}_num_switches"]+1) + features[f"{q2}_avg_cluster_size"]*(features[f"{q2}_num_switches"]+1)) / (features[f"{q1}_num_switches"] + features[f"{q2}_num_switches"] + 2)
        elif f"{q1}_avg_cluster_size" in features and f"{q1}_num_switches" in features:
            features[f"{feature_type}_avg_cluster_size"] = features[f"{q1}_avg_cluster_size"]
        elif f"{q2}_avg_cluster_size" in features and f"{q2}_num_switches" in features:
            features[f"{feature_type}_avg_cluster_size"] = features[f"{q2}_avg_cluster_size"]
        else:
            features[f"{feature_type}_avg_cluster_size"] = 0.0
        
        for metric in ["word_frequency_mean", "word_length_mean", "age_of_acquisition_mean"]:
            features[f"{feature_type}_{metric}"] = features.get(f"{q1}_{metric}", 0.0) * features.get(f"{q1}_{metric.replace('mean', 'total_words')}", 0) + features.get(f"{q2}_{metric}", 0.0) * features.get(f"{q2}_{metric.replace('mean', 'total_words')}", 0)
            features[f"{feature_type}_{metric.replace('mean', 'total_words')}"] = features.get(f"{q1}_{metric.replace('mean', 'total_words')}", 0) + features.get(f"{q2}_{metric.replace('mean', 'total_words')}", 0)
            if features[f"{feature_type}_{metric.replace('mean', 'total_words')}"] > 0:
                features[f"{feature_type}_{metric}"] /= features[f"{feature_type}_{metric.replace('mean', 'total_words')}"]

        features[f"{feature_type}_pause_rate"] = features.get(f"{q1}_pause_rate", 0.0) * features.get(f"{q1}_pause_rate_total_pauses", 0) + features.get(f"{q2}_pause_rate", 0.0) * features.get(f"{q2}_pause_rate_total_pauses", 0)
        features[f"{feature_type}_pause_rate_total_pauses"] = features.get(f"{q1}_pause_rate_total_pauses", 0) + features.get(f"{q2}_pause_rate_total_pauses", 0)
        if features[f"{feature_type}_pause_rate_total_pauses"] > 0:
            features[f"{feature_type}_pause_rate"] /= features[f"{feature_type}_pause_rate_total_pauses"]
        
        features[f"{feature_type}_speech_rate"] = features.get(f"{q1}_speech_rate", 0.0) * features.get(f"{q1}_speech_rate_total_time", 0) + features.get(f"{q2}_speech_rate", 0.0) * features.get(f"{q2}_speech_rate_total_time", 0)
        features[f"{feature_type}_speech_rate_total_time"] = features.get(f"{q1}_speech_rate_total_time", 0) + features.get(f"{q2}_speech_rate_total_time", 0)
        if features[f"{feature_type}_speech_rate_total_time"] > 0:
            features[f"{feature_type}_speech_rate"] /= features[f"{feature_type}_speech_rate_total_time"]
        
    return {k: [float(v)] for k, v in features.items()}

if __name__ == "__main__":
    # Load the wordnet lemmatizer for use in the neighborhood density function
    # Loading the clustering data
    nltk.download('wordnet')
    nltk.download("punkt")
    wordnet_lemmatizer = nltk.WordNetLemmatizer()

    # Loading the clustering data
    with open("data/animal_groups.txt", "r") as file:
        ac = file.readlines()
        animal_groups = {}
        animals = set()
        for line in ac:
            group_id, group_animals = line.strip().split(":")
            animal_groups[group_id] = group_animals.split(",")
            animal_groups[group_id] = [wordnet_lemmatizer.lemmatize(animal.strip().replace("_", " ")) for animal in animal_groups[group_id]]
            animals.update(animal_groups[group_id])
            
    with open("data/vegetable_groups.txt", "r") as file:
        vc = file.readlines()
        vegetable_groups = {}
        vegetables = set()
        for line in vc:
            group_id, group_vegetables = line.strip().split(":")
            vegetable_groups[group_id] = group_vegetables.split(",")
            vegetable_groups[group_id] = [wordnet_lemmatizer.lemmatize(veg.strip().replace("_", " ")) for veg in vegetable_groups[group_id]]
            vegetables.update(vegetable_groups[group_id])
    

    # Loading the response data from the input folder
    features_df = None
    total_files = len([file for file in os.listdir(input_dir) if file.endswith(".json")])
    for file in tqdm(os.listdir(input_dir), total=total_files):
        if file.endswith(".json"):
            match = re.match(r"RWRAD_(\d+).*", file)
            if match is not None:
                p_id = match.group(1)
            else:
                continue
            with open(os.path.join(input_dir, file), "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {file}. Skipping this file.")
                    continue
            features = process_data(data["responses"], aoa_path, aoa_sec_path, {"animal_groups": animal_groups, "vegetable_groups": vegetable_groups, "animal": animals, "vegetable": vegetables})
            features["patient_id"] = p_id
            if features_df is None:
                features_df = pd.DataFrame(features)
            else:
                features_df = pd.concat([features_df, pd.DataFrame(features)], ignore_index=True)
    
    # Save the features to a csv file
    if isinstance(features_df, pd.DataFrame):
        features_df = features_df[["patient_id"] + sorted([col for col in features_df.columns if col != "patient_id"])]
        features_df.to_csv(output_path, index=False)
    
    # We would also discretize the dataset by binning it into three bins with increasing thresholds based on the distribution of the features in the dataset and save it as a separate csv file
    if isinstance(features_df, pd.DataFrame):
        binned_features_df = features_df.copy()
        for col in binned_features_df.columns:
            if col != "patient_id":
                binned_features_df[col] = pd.qcut(binned_features_df[col], q=3, labels=False, duplicates="drop")
        binned_features_df.to_csv(binned_output_path, index=False)
    
