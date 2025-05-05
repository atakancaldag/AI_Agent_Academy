# Ai Agent Projesi

Bu proje, kullanıcıların sorularına en uygun cevapları sağlamak amacıyla **OpenAI Embeddings** kullanarak bir yapay zeka ajanı geliştirmektedir. Proje, kullanıcıdan alınan soruyu, önceden yüklenmiş verilerle karşılaştırarak en benzer soruyu bulur ve uygun cevabı döner.

## Özellikler

- **Veri Yükleme ve Ön İşleme**: Sorular ve cevaplar bir JSON dosyasından yüklenir.
- **Embedding Kullanımı**: Sorular için **OpenAI Embeddings** kullanılarak vektör temsilleri oluşturulur.
- **Benzerlik Hesaplama**: Kullanıcıdan alınan sorgu ile veritabanındaki sorular arasındaki benzerlik, **cosine similarity** algoritmasıyla hesaplanır.
- **Önbellek Kullanımı**: Oluşturulan embedding'ler, tekrar hesaplama gereksinimini önlemek için bir dosyada saklanır.

## Kullanım

1. **Kütüphanelerin Yüklenmesi**:\
   Proje, gerekli Python kütüphanelerini yüklemek için `requirements.txt` dosyasına sahiptir. Kütüphanelerini yüklemek için aşağıdaki komutu çalıştırın:
   
   ```bash
   pip install -r requirements.txt
1. **API Anahtarını Ayarlama**:\
   OpenAI API'sini kullanabilmek için, proje dizinine 'api_key.py' adında bir dosya oluşturup içine aşağıdaki gibi bir API anahtarı ekleyin veya 'main.py' dosyasındaki 'API_ANAHTARINIZI_BURAYA_GİRİN' kısmını kendi anahtarınızla doldurun:
      ```bash
      SECRET_KEY = 'API_ANAHTARINIZ'
   
1. **Projenin Çalıştırılması**:\
   Projeyi çalıştırmak için aşağıdaki komutu kullanabilirsiniz:
   
   ```bash
   python main.py
