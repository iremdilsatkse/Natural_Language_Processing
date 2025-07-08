"""
Metin üretimi gerçekleştirilecek.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# veri seti oluştur
text={
    "Kitap okumak beni gerçekten mutlu ediyor",
    "Bugün hava çok güzel, dışarıda yürüyüş yapmayı düşünüyorum",
    "Kahvemi alıp sahilde oturmak istiyorum",
    "Bugün biraz yorgunum, evde kalmak daha iyi olacak",
    "Arkadaşlarımla sinemaya gitmek harika olurdu",
    "Yeni bir diziye başladım, çok sürükleyici",
    "Yemek yapmayı seviyorum, özellikle makarna",
    "Müzik dinlemek moralimi yerine getiriyor",
    "Spor salonuna gitmek biraz zor geliyor ama şart",
    "Hafta sonu piknik yapmayı planlıyoruz",
    "Bu sabah işe geç kaldım, çok yoğundu trafik",
    "Kahvaltı günün en önemli öğünü bence",
    "Yeni telefonumun kamerası çok iyi çekiyor",
    "Fotoğraf çekmeyi her zaman sevdim",
    "Bugün çok kitap okudum, zihnim dinlendi",
    "Sessiz bir ortamda çalışmak bana iyi geliyor",
    "Akşam yemeğinde ne yapsam bilemiyorum",
    "Film izlemekten daha iyi bir rahatlama yöntemi yok",
    "Temiz hava almak için parka çıkacağım",
    "Arkadaşlarımla kahve içip sohbet etmeyi seviyorum",
    "Bugün güne enerjik başladım",
    "Evde yalnız kalmak bazen iyi geliyor",
    "Yeni tarifler denemek çok eğlenceli",
    "Sabah koşusu yapmak bana iyi hissettiriyor",
    "Gün batımını izlemek huzur verici",
    "Kedimle vakit geçirmek çok keyifli",
    "Yarın için plan yapmam gerekiyor",
    "Çalışma ortamımın düzenli olması beni motive ediyor",
    "Bazen müzik açıp dans etmek rahatlatıcı oluyor",
    "Bugün alışverişe çıkacağım",
    "Kütüphanede ders çalışmak daha verimli oluyor",
    "Bahçede çiçeklerle ilgilenmek beni sakinleştiriyor",
    "Yabancı dil öğrenmek istiyorum",
    "Online eğitimlere katılmak faydalı oluyor",
    "Yarın havanın yağmurlu olacağı söylendi",
    "Sevdiğim kitabı tekrar okumak istiyorum",
    "Bugün çok yoğun bir gün geçirdim",
    "Kardeşimle birlikte oyun oynamak çok eğlenceli",
    "Yeni insanlar tanımak beni heyecanlandırıyor",
    "Kahve kokusu sabahları bana enerji veriyor",
    "Yoga yapmak zihnimi boşaltmama yardımcı oluyor",
    "Bugün dışarı çıkmadan film maratonu yapacağım",
    "Sessizlik bazen en iyi müzik olabiliyor",
    "Kendi kendime konuşmak alışkanlık oldu",
    "Pijamalarla bütün gün evde kalmak güzel hissettiriyor",
    "Yeni bir hobi edinmek istiyorum",
    "Sosyal medyadan biraz uzak kalmak iyi gelebilir",
    "Sabahları erken kalkmak zor ama faydalı",
    "Bugün kendime zaman ayırmak istiyorum",
    "Güneşli havalarda daha pozitif oluyorum",
    "Çay demleyip cam kenarında oturmak huzur veriyor",
    "Bugün kendime küçük bir ödül vermek istiyorum",
    "Yeni bir şeyler öğrenmek beni her zaman heyecanlandırır",
    "Kitapçıda saatlerce vakit geçirmek çok keyifli",
    "Gece yürüyüşleri beni huzurlu hissettiriyor",
    "Küçük mutluluklar hayatı güzelleştiriyor",
    "Evimi yeniden dekore etmeyi düşünüyorum",
    "Sabah güne kahvaltısız başlamam",
    "Sevdiğim bir şarkıyı tekrar tekrar dinlemeyi seviyorum",
    "Bugün dışarıda arkadaşlarımla buluşacağım",
    "Akşam çayı içmeden günü tamamlayamıyorum",
    "Telefonumu bir süre sessize almak iyi gelecek",
    "Yeni ayakkabılarımı giymek için sabırsızlanıyorum",
    "Bazen hiçbir şey yapmadan durmak istiyorum",
    "Kalabalıklardan uzak olmak bana iyi geliyor",
    "Gece geç saatlerde kitap okumayı seviyorum",
    "Bugün çok üretken bir gün geçirdim",
    "Doğada vakit geçirmek beni tazeliyor",
    "Eski fotoğraflara bakmak nostalji yaratıyor",
    "Yeni bir defter alıp yazmaya başlamak istiyorum",
    "Akşamları sessizliği dinlemek beni dinlendiriyor",
    "Kahveyle birlikte çikolata yemeye bayılıyorum",
    "Arkadaşlarım beni motive eden insanlardır",
    "Bugün ruh halim oldukça dengeliydi",
    "Küçük şeylerle mutlu olmayı öğrendim",
    "Yarın spor yapmaya başlamayı düşünüyorum",
    "Bugün yeni tarifler denemek istiyorum",
    "Her sabah aynı kahvaltıyı yapıyorum",
    "Evde film gecesi planladık",
    "Yeni bir bitki aldım, adını henüz koymadım",
    "Yürüyüş yaparken podcast dinlemeyi seviyorum",
    "Hafta içi erken yatmak zor oluyor",
    "Rüyalarım son zamanlarda çok ilginç",
    "Bugün uzun zamandır görmediğim biriyle konuştum",
    "Koltukta uyuyakaldım, biraz boynum ağrıyor",
    "Kütüphaneye gitmeyi uzun zamandır planlıyorum",
    "Telefonumu elimden bırakmakta zorlanıyorum",
    "Yeni bir dizinin fragmanını izledim, ilgimi çekti",
    "Sessiz filmleri izlemek farklı bir deneyim sunuyor",
    "Bugün beklenmedik güzel haberler aldım",
    "Uzun bir banyo yaparak rahatladım",
    "Yaz aylarını kıştan daha çok seviyorum",
    "Yalnız kalınca daha verimli çalışabiliyorum",
    "Yeni tarif defterime birkaç not daha ekledim",
    "Günün sonunda günlüğüme yazmak alışkanlığım oldu",
    "Bugün yoğun ama verimli bir gündü",
    "Arkadaşım bana sürpriz yaptı, çok mutlu oldum",
    "Pazartesi günleri biraz zorlayıcı oluyor",
    "Evde küçük değişiklikler yapmak bile fark yaratıyor",
    "Bugün kendime vakit ayırabildim",
    "Bugün dışarıda yürüyüş yaparken eski bir arkadaşıma rastladım",
    "Yeni aldığım defteri yazmak için sabırsızlanıyorum",
    "Evde yalnız kaldığımda müzik açmak hoşuma gidiyor",
    "Günün en sevdiğim anı sıcak çayımla kitap okumak",
    "Balkonda oturup sessizliği dinlemek çok rahatlatıcı",
    "Yeni bir diziye başlamak için akşamı bekliyorum",
    "Kedim bütün gün kucağımda uyudu, çok sevimliydi",
    "Bugün pazardan taze meyveler aldım",
    "Bir fincan kahveyle güne başlamak harika bir his",
    "Dışarıda yağmur var ama evde olmanın huzuru başka",
    "Sabah alarm çalınca kalkmakta zorlandım",
    "Kütüphanede ders çalışmak bana ilham veriyor",
    "Yeni bir şey öğrenmek beni heyecanlandırıyor",
    "Bugün sevdiğim bir yemeği yapacağım",
    "Pencere kenarında oturup dışarıyı izlemek güzel",
    "Planlı çalışınca daha az strese giriyorum",
    "Yürürken düşüncelerimle baş başa kalıyorum",
    "Telefonumun şarjı bitince kendime daha çok vakit ayırıyorum",
    "Arkadaşımın doğum günü için hediye baktım",
    "Bugün çok güzel bir rüya gördüm",
    "Kahve içerken notlarımı gözden geçirdim",
    "Sabah erken kalkmak başlangıçta zor ama gün içinde avantajlı",
    "Hava kapalı olunca enerjim biraz düşüyor",
    "Bazen sadece sessiz bir ortamda oturmak istiyorum",
    "Bugün kendime bir mola verdim",
    "Yarın için yapılacaklar listemi hazırladım",
    "Eski bir şarkıyı yeniden dinlemek anılarımı canlandırdı",
    "Güne güzel başlamak için erken yatmak istiyorum",
    "Bugün annemle uzun bir telefon konuşması yaptım",
    "Biraz yürüyüş yapmak iyi hissettirdi",
    "Ders çalışırken zamanın nasıl geçtiğini anlamıyorum",
    "Kitapçıdan çıkmak istemedim, hepsi çok güzeldi",
    "Yemek sonrası tatlı yemek alışkanlık haline geldi",
    "Bugün notlarımı düzenledim, artık her şey daha net",
    "Pencereye vuran yağmur sesi çok hoşuma gidiyor",
    "Hafta sonu için kısa bir kaçamak planlıyorum",
    "Yeni bir çizim programı keşfettim ve çok sevdim",
    "Bugün dışarı çıkmak yerine evde kalmayı tercih ettim",
    "Arkadaşım bana moral verdi, iyi hissettim",
    "Taze çiçekler odaya bambaşka bir hava katıyor",
    "Her gün biraz meditasyon yapmaya çalışıyorum",
    "Bugün dışarıda harika bir gün batımı vardı",
    "Evde temizlik yapmak terapi gibi geliyor",
    "Kendi yaptığım playlistleri dinlemek beni mutlu ediyor",
    "Kafamı dağıtmak için kısa bir yürüyüş yaptım",
    "Sabahları günlüğüme yazı yazarak başlıyorum",
    "Bugün denediğim tarif çok başarılı oldu",
    "Kardeşimle uzun süredir bu kadar çok gülmemiştik",
    "Yeni projelere başlamak için motivasyon topluyorum",
    "Gün sonunda sıcak bir duş bütün yorgunluğumu alıyor",
    "Bugün balkonda oturup sıcak çay içmek istiyorum",
    "Kütüphanede vakit geçirmek beni çok motive ediyor",
    "Akşamları gün batımını izlemek huzur veriyor",
    "Kahve makinem bozuldu, sabahlar eksik kaldı",
    "Yeni dizimin bölümü yayınlandı, çok heyecanlıyım",
    "Arkadaşım bana kitap önerdi, hemen sipariş verdim",
    "Bugün biraz da olsa temiz hava almak istedim",
    "Yarın erken uyanıp güzel bir kahvaltı yapacağım",
    "Yeni ajandam geldi, plan yapmaya başladım",
    "Akşam yürüyüşleri zihnimi toparlamama yardımcı oluyor",
    "Bugün kendime sakin bir gün armağan ettim",
    "Film izlerken battaniyeye sarınmak ayrı bir keyif",
    "Evde kek pişirmenin kokusu her yeri sardı",
    "Kitapçıdan çıkarken elim dolu dolu oluyor",
    "Sabah sessizliği bana huzur veriyor",
    "Bugün uzun süredir görüşmediğim biriyle buluştum",
    "Kendime yeni hedefler koymak beni motive ediyor",
    "Akşam oturup günümü değerlendirmeyi seviyorum",
    "Küçük mutlulukları fark etmek günümü güzelleştiriyor",
    "Bugün dışarısı çok kalabalıktı, biraz bunaldım",
    "Telefonumu sessize almak iyi bir fikir oldu",
    "Bugün spor yapmaya üşendim ama sonradan iyi geldi",
    "Camı açınca içeri giren serin hava iyi hissettirdi",
    "Kahve molası günün en keyifli anlarından biri",
    "Gün sonunda kitap okumak dinlendirici oluyor",
    "Yeni aldığım çiçek odaya çok güzel renk kattı",
    "Bugün sadece dinlenmek istedim, hiçbir şey yapmadım",
    "Hafta sonu için küçük bir seyahat planlıyorum",
    "Arkadaşım bana güzel bir mektup yazmış",
    "Bugün dolabı düzenledim, çok ferahladım",
    "Kendi kendime konuşurken yakaladım kendimi",
    "Dışarıda yürürken hafif bir esinti vardı",
    "Çalışma masamı sadeleştirmek iyi hissettirdi",
    "Bu sabah kahvaltı ederken müzik dinledim",
    "Uzun zamandır yazmak istediğim şeyleri yazdım",
    "Bugün camları açıp evi havalandırdım",
    "Kahvemi alıp pencere önüne geçtim",
    "Gün içinde biraz sessizlik bana iyi geliyor",
    "Bugün her şey yolunda gitti, şükrettim",
    "Yeni bir kitaplık almak istiyorum",
    "Bugün pijamalarımla dolaştım, rahat bir gündü",
    "Pazartesiye güzel başlamak için erkenden hazırlandım",
    "Dışarı çıkmadan önce hava durumuna baktım",
    "Bazen eski mesajlara bakmak beni duygulandırıyor",
    "Bugün yeni bir müzik listesi oluşturdum",
    "Pencereden dışarı bakarken yağmur başladı",
    "Kahvemi içerken defterime bir şeyler karaladım",
    "Evde mum yakmak ortama huzur katıyor",
    "Bugün içimden hiçbir şey yapmak gelmedi",
    "Yarın daha üretken olacağıma inanıyorum"
}

### Metin Temizleme ve Preprocessing
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for text in text:
    token_list = tokenizer.texts_to_sequences([text])[0]
    # ngram dizisi
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# en uzun dizi
max_sequence_len = max([len(x) for x in input_sequences])

# tüm dizilerin aynı uzunlukta olması sağlanır
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

X = input_sequences[:,:-1]
y = input_sequences[:,-1]

y = tf.keras.utils.to_categorical(y, num_classes=total_words) # one-hot encoding

### LSTM Modeli
model = Sequential()

model.add(Embedding(total_words, 50, input_length=X.shape[1]))

model.add(LSTM(100, return_sequences=False))

model.add(Dense(total_words, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X, y, epochs=100, verbose=1)

### model tahmini
def generate_text(seed_text, next_words):
    for _ in range(next_words):
        # input sayısal verilere dönüştürülür
        token_list = tokenizer.texts_to_sequences([seed_text])[0]

        # padding
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

        # tahmin
        predicted = model.predict(token_list, verbose = 0)

        # en yüksek olasılığa sahip kelimenin indexi 
        predicted_word_index = np.argmax(predicted, axis = -1)

        # tokenizer ile kelime indexinden asıl kelime bulunması
        predicted_word = tokenizer.index_word[predicted_word_index[0]]

        # tahmin edilen kelime
        seed_text = seed_text + " " + predicted_word
        
    return seed_text

seed_text = "Bu hafta sonu"

print(generate_text(seed_text, 5))