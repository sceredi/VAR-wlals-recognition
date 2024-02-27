from typing import List, Tuple


def analyze_landmarks(landmark) -> Tuple[List[List[float]], List[List[float]]]:
    handedness = landmark.handedness
    if len(handedness) == 0:
        return [], []
    hand_landmarks = landmark.hand_landmarks
    # hand_world_landmarks = landmark.hand_world_landmarks
    if len(handedness) == 1:
        return _get_one_hand(handedness, hand_landmarks)
    if len(handedness) == 2:
        return _get_two_hands(handedness, hand_landmarks)
    # print(f"hand_world_landmarks: {hand_world_landmarks}")
    print(f"A problem occourred, handedness is {len(handedness)}")
    return [], []

def _get_one_hand(handedness, hand_landmarks) -> Tuple[List[List[float]], List[List[float]]]:
    if handedness[0][0].category_name == "Right":
        return [], _hand_landmarks_to_list(hand_landmarks[0])
    else:
        return _hand_landmarks_to_list(hand_landmarks[0]), []

def _get_two_hands(handedness, hand_landmarks) -> Tuple[List[List[float]], List[List[float]]]:
    if handedness[0][0].category_name == "Left" and handedness[1][0].category_name == "Right":
        return _hand_landmarks_to_list(hand_landmarks[0]), _hand_landmarks_to_list(hand_landmarks[1])
    if handedness[0][0].category_name == "Right" and handedness[1][0].category_name == "Left":
        return _hand_landmarks_to_list(hand_landmarks[1]), _hand_landmarks_to_list(hand_landmarks[0])

    if handedness[0][0].category_name == "Right" and handedness[1][0].category_name == "Right":
        # Sometimes both categories are Right
        if handedness[0][0].score > handedness[1][0].score:
            # If the score of the first hand is higher, it is the right hand
            return _hand_landmarks_to_list(hand_landmarks[1]), _hand_landmarks_to_list(hand_landmarks[0])
        else:
            # Else the second hand is the right hand
            return _hand_landmarks_to_list(hand_landmarks[0]), _hand_landmarks_to_list(hand_landmarks[1])
    if handedness[0][0].category_name == "Left" and handedness[1][0].category_name == "Left":
        # Sometimes both categories are Left
        if handedness[0][0].score > handedness[1][0].score:
            # If the score of the first hand is higher, it is the left hand
            return _hand_landmarks_to_list(hand_landmarks[0]), _hand_landmarks_to_list(hand_landmarks[1])
        else:
            # Else the second hand is the left hand
            return _hand_landmarks_to_list(hand_landmarks[1]), _hand_landmarks_to_list(hand_landmarks[0])
    return [], []

def _hand_landmarks_to_list(hand_landmarks) -> List[List[float]]:
    return [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks]
