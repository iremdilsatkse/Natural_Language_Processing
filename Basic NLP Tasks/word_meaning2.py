from pywsd.lesk import simple_lesk, adapted_lesk, cosine_lesk

sentences = [
    "I got to the bank to deposite money",
    "The river bank wass flooded after the heavy rain"
]

word = "bank"

for s in sentences:
    print("Sentences:", s)

    sense_simple_lesk = simple_lesk(s, word)
    print("Sense simple:", sense_simple_lesk.definition())

    sense_adapted_lesk = adapted_lesk(s, word)
    print("Sense adapted:", sense_adapted_lesk.definition())

    sense_cosine_lesk = cosine_lesk(s, word)
    print("Sense cosine:", sense_cosine_lesk.definition())