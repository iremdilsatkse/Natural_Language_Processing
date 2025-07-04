# metinlerde bulunan fazla boşlukları kaldırma
text = "This   is   an   example   text."

text.split()  
cleaned_text1 = " ".join(text.split())
print(cleaned_text1) 


# büyük harf/küçük dönüşüm işlemleri
text = "This is an example text."

cleaned_text2 = text.lower() 
print(cleaned_text2)  


# metinlerdeki noktalama işaretlerini kaldırma
import string
text = "This is an example text."

cleaned_text3 = text.translate(str.maketrans("", "", string.punctuation))
print(cleaned_text3)


# metinlerdeki özel karakterleri kaldırma
import re
text = "This is an example text. @#$%^&*()"

cleaned_text4 = re.sub(r"[^a-zA-Z0-9\s]", "", text)
print(cleaned_text4)


# yazım hatalarını düzelt
from textblob import TextBlob
text = "This is an exomple text."

cleaned_text5 = TextBlob(text).correct()
print(cleaned_text5)


# html ya da url etiketlerin kaldırılması
from bs4 import BeautifulSoup
text = "<div>This is an example text.</div>"

cleaned_text6 = BeautifulSoup(text, "html.parser").get_text()
print(cleaned_text6)