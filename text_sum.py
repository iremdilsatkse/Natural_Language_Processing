from transformers import pipeline

summarizer = pipeline("summarization")

text = """
Ankara, the capital of Turkey, lies in the heart of Anatolia at an altitude of 850 meters. 
Known in ancient times as Ankyra (meaning “anchor”) or Ancyra, it was an important cultural, trading and arts center in the Roman period, and a trading hub on the caravan route to the east in the Ottoman era. 
Chosen as a base by Mustafa Kemal Atatürk during the War of Liberation, the city was declared the capital of the Republic of Turkey in 1923.
Today, Ankara is a modern city with a population of over 5 million, but it still retains the traces of civilizations dating back to the Bronze Age, to the Hittites, Phrygians, Lydians and Persians. 
The Romans and the Byzantines also left their marks in the region. 
The Museum of Anatolian Civilizations, located on a hill near the ancient citadel, houses a unique collection of treasures dating back to 2000 B.C.
On another imposing hill in the center of Ankara stands Kemal Atatürk’s mausoleum, the Anıtkabir, a fusion of ancient and modern architectural ideas.
"""

summary = summarizer(
    text,
    max_length = 90,
    min_length = 45,
    do_sample = True 
    )

print(summary[0]["summary_text"])

"""
Ankyra (meaning ‘anchor’) or Ancyra, was an important cultural, trading and arts center in the Roman period, and a trading hub on the caravan route to the east in the Ottoman era . 
The Museum of Anatolian Civilizations houses a unique collection of treasures dating back to 2000 B.C.
"""




