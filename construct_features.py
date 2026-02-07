# This is the file for constructing the features for the Alzheimer's speech dataset
import json
import pandas as pd
import wordfreq
import nltk
import pronouncing

nltk.download('wordnet')
nltk.download("punkt")

def word_frequency(response: list, aggregate: str="mean") -> float:
    # We would use the wordfreq library to calculate the frequency of words in the response
    """
    Calculate the average word frequency in a response.
    :param response: The processed response string
    :param aggregate: Specify "mean" for average word frequency or "total" for total frequency of all words
    :return: Mean word frequency or total frequency of all words
    """
    words = [r.strip().lower() for r in response]
    words = list(set(words))  # Consider unique words only
    frequencies = [wordfreq.word_frequency(word, "en") for word in words]
    if aggregate == "total":
        return sum(frequencies)
    elif aggregate == "mean":
        return sum(frequencies) / len(frequencies) if len(frequencies) > 0 else 0.0
    else:
        raise NotImplementedError(f"Aggregate method {aggregate} not implemented. Use 'mean' or 'total'.")

def word_length(response: list, aggregate: str="mean") -> float:
    """Calculate the length of an average word in a response.
    :param response: The processed response string
    :param aggregate: Specify "mean" for average word length or "total" for total length of all words
    :return: Mean word length or total length of all words
    """
    words = [r.strip().lower() for r in response]
    words = list(set(words))  # Consider unique words only
    lengths = [len(word) for word in words]
    if aggregate == "mean":
        return sum(lengths) / len(lengths) if lengths else 0.0
    elif aggregate == "total":
        return sum(lengths)
    else:
        raise NotImplementedError(f"Aggregate method {aggregate} not implemented. Use 'mean' or 'total'.")

def age_of_acquisition(response: list, aoa_path: str, aggregate: str="mean") -> float:
    """Calculate the average age of acquisition of words in a response.
    :param response: The processed response string
    :param aoa_path: Path to the age of acquisition data file
    :param aggregate: Specify "mean" for average AoA or "total" for total AoA of all words
    :return: Mean AoA or total AoA of all words
    """
    words = [r.strip().lower() for r in response]
    words = list(set(words))  # Consider unique words only
    # Use nltk to lemmatize words
    lemmatizer = nltk.WordNetLemmatizer()
    lemmas = {word: lemmatizer.lemmatize(word) for word in words}
    # Load age of acquisition data which is an xlsx file (Sheet1)
    aoa_data = pd.read_excel(aoa_path, sheet_name="Sheet1")
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
                    word_aoa[word] = None
                else:
                    word_aoa[word] = row["AoA_Kup_lem"].values[0]
            else:
                if not row["AoA_Kup"].isnull().values[0]:
                    word_aoa[word] = row["AoA_Kup"].values[0]
                elif not row["AoA_Kup_lem"].isnull().values[0]:
                    word_aoa[word] = row["AoA_Kup_lem"].values[0]
                else:
                    word_aoa[word] = None
        else:
            if not row["AoA_Kup"].isnull().values[0]:
                word_aoa[word] = row["AoA_Kup"].values[0]
            elif not row["AoA_Kup_lem"].isnull().values[0]:
                word_aoa[word] = row["AoA_Kup_lem"].values[0]
            else:
                word_aoa[word] = None
    # Now calculate the aggregate AoA
    aoa_values = [aoa for aoa in word_aoa.values() if aoa is not None]
    if not aoa_values:
        return 0.0
    if aggregate == "mean":
        return sum(aoa_values) / len(aoa_values)
    elif aggregate == "total":
        return sum(aoa_values)
    else:
        raise NotImplementedError(f"Aggregate method {aggregate} not implemented. Use 'mean' or 'total'.")

def neigborhood_density(response: list, clustering_type: str="semantic", **kwargs) -> dict:
    words = [r.strip().lower() for r in response]
    if clustering_type == "semantic":
        # We would consider the list of words generated from the response and create clusters on consecutive words based on
        # whether they 
        if kwargs and kwargs["animals"] and kwargs["vegetables"]:
            # Is the list of words provided animals or vegetables (Word list should only contain one of these)
            for word in words:
                if word in kwargs["animals"]:
                    current_cluster = "animal"
                    break
                elif word in kwargs["vegetables"]:
                    current_cluster = "vegetable"
                    break
                else:
                    continue
            # Once we have the current cluster
            group_dict = kwargs[f"{current_cluster}_groups"]
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
                        clusters[current_cluster_id].append(word_in_cluster)
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

def pause_rate(pauses, pause_threshold_in_seconds: float, aggregate: str="mean") -> float:
    """
    Calculate pause features from the list of pauses.
    :param pauses: List of pauses with their start and end times.
    :param pause_threshold_in_seconds: Threshold to consider a pause significant. Pauses lower than the thtshold are ignored.
    :param aggregate: Description of the aggregate method to use ("mean" or "total")
    :return: The calculated pause feature based on the specified aggregate method.
    """
    # The pauses are a list of dictionaries with pause start and end times
    for pause in pauses:
        pause_duration = float(pause["end"]) - float(pause["start"])
        pause["duration"] = pause_duration
    significant_pauses = [pause["duration"] for pause in pauses if pause["duration"] >= pause_threshold_in_seconds]
    if not significant_pauses:
        return 0.0
    if aggregate == "mean":
        return sum(significant_pauses) / len(significant_pauses)
    elif aggregate == "total":
        return sum(significant_pauses)
    else:
        raise NotImplementedError(f"Aggregate method {aggregate} not implemented. Use 'mean' or 'total'.")

def speech_rate(raw_response, time_segments) -> float:
    """
    Calculate speech rate as words per second.
    :param raw_response: The raw response text from which we can calculate the total number of words spoken
    :param time_segments: The list of time segments corresponding to the responses
    :return: The speech rate in words per second
    """
    words = nltk.word_tokenize(raw_response)
    words = [word for word in words if word.isalnum()]  # Consider only alphanumeric tokens as words
    total_words = len(words)

    total_time = sum(float(segment["end"]) - float(segment["start"]) for segment in time_segments)
    if total_time == 0:
        return 0.0
    return total_words / total_time

def process_data(response_data: dict, aoa_path: str, clustering_data: dict) -> dict:
    """
    Process the response data to extract features.
    :param response_data: The processed response data for a patient
    :param aoa_path: Path to the age of acquisition data file
    :param clustering_data: The data required for clustering (e.g., animal and vegetable groups)
    :return: A dictionary containing the extracted features
    """
    # The dictionary has keys, R1, R2, r3 and 4r4 corresponding to letter f ,letter L, Animals and vegetables respectively
    features = {}

    # Process for R1 and R2 which are letter f and letter L respectively
    for response_key in ["R1", "R2"]:
        if response_key in response_data and response_data[response_key]:
            features[f"{response_key}_word_frequency_mean"] = word_frequency(response_data[response_key]["extracted_answer"], aggregate="mean")
            features[f"{response_key}_word_length_mean"] = word_length(response_data[response_key]["extracted_answer"], aggregate="mean")
            features[f"{response_key}_age_of_acquisition_mean"] = age_of_acquisition(response_data[response_key]["extracted_answer"], aoa_path, aggregate="mean")
            cluster_metrics = neigborhood_density(response_data[response_key]["extracted_answer"], clustering_type="phonetic")
            features[f"{response_key}_num_switches"] = cluster_metrics["num_switches"]
            features[f"{response_key}_avg_cluster_size"] = cluster_metrics["avg_cluster_size"]
            features[f"{response_key}_total_words"] = cluster_metrics["total_words"]
            features[f"{response_key}_pause_rate_mean"] = pause_rate(response_data[response_key]["pauses"], pause_threshold_in_seconds=0.5, aggregate="mean")
            features[f"{response_key}_speech_rate"] = speech_rate(response_data[response_key]["full_response"], response_data[response_key]["response_timestamps"])
    
    for response_key in ["R3", "R4"]:
        if response_key in response_data and response_data[response_key]:
            features[f"{response_key}_word_frequency_mean"] = word_frequency(response_data[response_key]["extracted_answer"], aggregate="mean")
            features[f"{response_key}_word_length_mean"] = word_length(response_data[response_key]["extracted_answer"], aggregate="mean")
            features[f"{response_key}_age_of_acquisition_mean"] = age_of_acquisition(response_data[response_key]["extracted_answer"], aoa_path, aggregate="mean")
            cluster_metrics = neigborhood_density(response_data[response_key]["extracted_answer"], clustering_type="semantic", **clustering_data)
            features[f"{response_key}_num_switches"] = cluster_metrics["num_switches"]
            features[f"{response_key}_avg_cluster_size"] = cluster_metrics["avg_cluster_size"]
            features[f"{response_key}_total_words"] = cluster_metrics["total_words"]
            features[f"{response_key}_pause_rate_mean"] = pause_rate(response_data[response_key]["pauses"], pause_threshold_in_seconds=0.5, aggregate="mean")
            features[f"{response_key}_speech_rate"] = speech_rate(response_data[response_key]["full_response"], response_data[response_key]["response_timestamps"])
    
    return features

if __name__ == "__main__":
    # Load the sample file
    print("Loading processed response data...")
    with open("/home/rxs174730/programming/speech/processed_response.json", "r") as file:
        data = json.load(file)
    
    # Loading the clustering data
    with open("data/animal_groups.txt", "r") as file:
        ac = file.readlines()
        animal_groups = {}
        animals = set()
        for line in ac:
            group_id, group_animals = line.strip().split(":")
            animal_groups[group_id] = group_animals.split(",")
            animals.update(animal_groups[group_id])
            
    with open("data/vegetable_groups.txt", "r") as file:
        vc = file.readlines()
        vegetable_groups = {}
        vegetables = set()
        for line in vc:
            group_id, group_vegetables = line.strip().split(":")
            vegetable_groups[group_id] = group_vegetables.split(",")
            vegetables.update(vegetable_groups[group_id])
    
    process_data(data["responses"], "data/age_of_acquisition.xlsx", {"animal_groups": animal_groups, "vegetable_groups": vegetable_groups})
    
    
