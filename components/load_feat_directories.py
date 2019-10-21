
# config = 0 ==> CS servers, config = 1 ==> DESK038, config = 2 ==> MACBOOK, config = 3 ==> Rivanna

config = 1

if config == 0:                                 # CS Server
    prefix = '..//..//..//Datasets//'
elif config == 1:                               # DESKTOP 038
    prefix = 'D://Datasets//TRAINING//'
elif config == 2:                               # MACBOOK
    prefix = '//Users//yegao//Documents//Datasets//'
else:                                           # Rivanna
    prefix = '..//..//..//Datasets//'

allnoised_happy = prefix + 'padded_deamplified_allnoised_reverberated//Happy//'
allnoised_happy_npy = prefix + \
    'padded_deamplified_allnoised_reverberated//npy//Happy_npy//'
allnoised_happy_npy_test = prefix + \
    'padded_deamplified_allnoised_reverberated//npy//Happy_npy_test//'


allnoised_angry = prefix + 'padded_deamplified_allnoised_reverberated//Angry//'
allnoised_angry_npy = prefix + \
    'padded_deamplified_allnoised_reverberated//npy//Angry_npy//'
allnoised_angry_npy_test = prefix + \
    'padded_deamplified_allnoised_reverberated//npy//Angry_npy_test//'


allnoised_neutral = prefix + 'padded_deamplified_allnoised_reverberated//Neutral//'
allnoised_neutral_npy = prefix + \
    'padded_deamplified_allnoised_reverberated//npy//Neutral_npy//'
allnoised_neutral_npy_test = prefix + \
    'padded_deamplified_allnoised_reverberated//npy//Neutral_npy_test//'


allnoised_sad = prefix + 'padded_deamplified_allnoised_reverberated//Sad//'
allnoised_sad_npy = prefix + \
    'padded_deamplified_allnoised_reverberated//npy//Sad_npy//'
allnoised_sad_npy_test = prefix + \
    'padded_deamplified_allnoised_reverberated//npy//Sad_npy_test//'


allnoised_other = prefix + 'padded_deamplified_allnoised_reverberated//Other//'
allnoised_other_npy = prefix + \
    'padded_deamplified_allnoised_reverberated//npy//Other_npy//'
allnoised_other_npy_test = prefix + \
    'padded_deamplified_allnoised_reverberated//npy//Other_npy_test//'


homenoised_happy = prefix + 'padded_deamplified_homenoised_reverberated//Happy//'
homenoised_happy_npy = prefix + \
    'padded_deamplified_homenoised_reverberated//npy//Happy_npy//'
homenoised_happy_npy_test = prefix + \
    'padded_deamplified_homenoised_reverberated//npy//Happy_npy_test//'


homenoised_angry = prefix + 'padded_deamplified_homenoised_reverberated//Angry//'
homenoised_angry_npy = prefix + \
    'padded_deamplified_homenoised_reverberated//npy//Angry_npy//'
homenoised_angry_npy_test = prefix + \
    'padded_deamplified_homenoised_reverberated//npy//Angry_npy_test//'


homenoised_neutral = prefix + 'padded_deamplified_homenoised_reverberated//Neutral//'
homenoised_neutral_npy = prefix + \
    'padded_deamplified_homenoised_reverberated//npy//Neutral_npy//'
homenoised_neutral_npy_test = prefix + \
    'padded_deamplified_homenoised_reverberated//npy//Neutral_npy_test//'


homenoised_sad = prefix + 'padded_deamplified_homenoised_reverberated//Sad//'
homenoised_sad_npy = prefix + \
    'padded_deamplified_homenoised_reverberated//npy//Sad_npy//'
homenoised_sad_npy_test = prefix + \
    'padded_deamplified_homenoised_reverberated//npy//Sad_npy_test//'

homenoised_other = prefix + 'padded_deamplified_homenoised_reverberated//Other//'
homenoised_other_npy = prefix + \
    'padded_deamplified_homenoised_reverberated//npy//Other_npy//'
homenoised_other_npy_test = prefix + \
    'padded_deamplified_homenoised_reverberated//npy//Other_npy_test//'


# allnoised = [allnoised_happy, allnoised_angry, allnoised_neutral, allnoised_sad, allnoised_other]

allnoised_npy = [allnoised_happy_npy, allnoised_angry_npy,
                 allnoised_neutral_npy, allnoised_sad_npy, allnoised_other_npy]

allnoised_npy_test = [allnoised_happy_npy_test, allnoised_angry_npy_test,
                 allnoised_neutral_npy_test, allnoised_sad_npy_test, allnoised_other_npy_test]

# homenoised = [homenoised_happy, homenoised_angry, homenoised_neutral, homenoised_sad, homenoised_other]

homenoised_npy = [homenoised_happy_npy, homenoised_angry_npy,
                  homenoised_neutral_npy, homenoised_sad_npy, homenoised_other_npy]

homenoised_npy_test = [homenoised_happy_npy_test, homenoised_angry_npy_test,
                  homenoised_neutral_npy_test, homenoised_sad_npy_test, homenoised_other_npy_test]

# allnoised_npy[0, 1, 2, 3, 4] ==> H, A, N, S, O
# homenoised_npy[0, 1, 2, 3, 4] ==> H, A, N, S, O
