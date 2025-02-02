import boto3
import json
import logging
from botocore.exceptions import ClientError
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from dotenv import load_dotenv

load_dotenv()


APIKEY = os.getenv("API_KEY")
ACCESSKEY = os.getenv("ACCESSKEY")
ELEVENLABS_APIKEY = os.getenv("ELEVENLABS_APIKEY")
os.environ['ELEVENLABS_APIKEY'] = ELEVENLABS_APIKEY
os.environ['AWS_ACCESS_KEY_ID'] = ACCESSKEY
os.environ['AWS_SECRET_ACCESS_KEY'] = APIKEY
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

SERVICE_NAME = 'bedrock-runtime'
REGION_NAME = 'us-east-1'
ANTHROPIC_VERSION = 'bedrock-2023-05-31'
TEMPERATURE = 0.5
MAX_TOKENS = 3000

def retrieve_chunks_from_kb(query, knowledge_base_id):
    """
    İlgili sorguya göre knowledge base’den parçaları getirir.
    """
    try:
        bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name=REGION_NAME)
        retrieval_request = {
            'knowledgeBaseId': knowledge_base_id,
            'retrievalQuery': {'text': query},
            'retrievalConfiguration': {
                'vectorSearchConfiguration': {'numberOfResults': 10}
            }
        }
        response = bedrock_agent_runtime.retrieve(**retrieval_request)
        retrieval_results = response.get('retrievalResults', [])
        retrieved_chunks = [result['content']['text'] for result in retrieval_results if 'text' in result['content']]
        return retrieved_chunks

    except ClientError as err:
        error_message = err.response["Error"]["Message"]
        logger.error(f"Retrieval sırasında client hatası oluştu: {error_message}")
        return []

def generate_response_with_llm(prompt, model_id='us.anthropic.claude-3-5-haiku-20241022-v1:0'):
    """
    Verilen prompt’u belirtilen LLM modeline gönderip yanıtı döner.
    """
    try:
        bedrock_runtime = boto3.client(service_name=SERVICE_NAME, region_name=REGION_NAME)
        body = json.dumps({
            'anthropic_version': ANTHROPIC_VERSION,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': MAX_TOKENS,
            'temperature': TEMPERATURE,
        })
        response = bedrock_runtime.invoke_model(body=body, modelId=model_id)
        response_body = json.loads(response.get('body').read())
        return response_body.get('content')[0].get('text')
    except ClientError as err:
        error_message = err.response["Error"]["Message"]
        logger.error(f"Model çağrısı sırasında client hatası oluştu: {error_message}")
        return "An error occurred while generating the response."

def get_video_script_and_quiz(user_query):
    """
    Kullanıcının sorusuna göre:
    1. Knowledge base’den ilgili belgeleri getirir,
    2. Konu özetini oluşturup 180 saniyelik video segmentleri üretir (video script JSON olarak),
    3. Oluşan video scripti kullanarak; segmentleri 1–3, 4–6 ve 7–10 aralıklarına göre quiz soruları (JSON formatında) oluşturur.
    Sonuç olarak, video script ve quiz çıktısını döner.
    """
    knowledge_base_id = 'QALSFMRFUA'  # Gerçek KB ID’nizi girin
    query_improve_prompt = (
        "Verilen soruyu analiz ederek retrieval sürecinde kullanılabilecek, "
        "konuyla ilgili anahtar noktaları, alt konuları ve detayları içeren, "
        "sadece hipotetik belgeyi çıktı olarak verecek şekilde bir belge taslağı oluştur.\n\n"
        "Aşağıda analiz edilecek orijinal soru verilmiştir:\n\n"
        f"Soru: {user_query}\n\n"
        "Lütfen yalnızca retrieval için kullanılacak örnek, hipotetik belgeyi oluştur ve başka hiçbir ek bilgi vermeden çıktı olarak sadece belgeyi sun."
    )
    # İyileştirilmiş sorguyu ve hipotetik belgeyi üret
    improved_query = generate_response_with_llm(query_improve_prompt)
    retrieved_chunks = retrieve_chunks_from_kb(improved_query, knowledge_base_id)

    if retrieved_chunks:
        # Belirlenen parçalardan tek bir bağlam oluşturulur.
        context = "\n".join(retrieved_chunks)
        summary_prompt = (
            "Ders kitabından alınan bağlamı kullanarak, aşağıdaki soruyu detaylı, kapsamlı ve anlaşılır şekilde açıklayan "
            "bir konu özeti oluştur. Lütfen açıklamanızda:\n"
            "- Konunun temel noktalarını ve alt başlıklarını belirgin şekilde vurgulayın,\n"
            "- Gerekirse örnekler ve açıklamalarla destekleyin,\n"
            "- Mantıksal bir akış ve yapı izleyerek konuyu parçalara bölün.\n\n"
            f"Bağlam:\n{context}\n\n"
            f"Soru:\n{user_query}\n\n"
            "Özet:"
        )
        summary = generate_response_with_llm(summary_prompt)

        video_script_prompt = (
            "Ders kitabından alıp oluşturduğumuz konu özetini kullanarak, toplam 180 saniyelik bir video oluştur. "
            "Videonun her bir segmenti aşağıdaki gereksinimlere uygun olsun:\n\n"
            "1. Toplam süre 180 saniye olacak şekilde, segment süreleri 10 ila 20 saniye arasında ayarlanmış ve ardışık zaman dilimlerine bölünmüş olsun.\n"
            "2. Her segmentin başlangıç ve bitiş zamanları birbirine bitişik olmalı (örneğin, bir segmentin 'end_time'ı 10 ise, "
            "bir sonraki segmentin 'start_time'ı 10 olmalı).\n"
            "3. Her segmentte aşağıdaki anahtarlar yer almalıdır:\n"
            "   - \"video_prompt\": Video oluşturma için sahne tanımlamaları, görsel detaylar vb. içeren prompt,\n"
            "   - \"video_script\": Text-to-speech için kullanılacak anlatım metni.\n\n"
            "4. Video scriptleri, konunun temel kavramlarını ve önemli detaylarını içerecek şekilde açık, net, "
            "öğretici ve akıcı bir dilde hazırlanmalıdır.\n\n"
            "Örnek JSON formatı:\n"
            "[\n"
            "  {\n"
            "    \"video_prompt\": \"Example prompt text for the first segment\",\n"
            "    \"video_script\": \"Example script text for the first segment\",\n"
            "    \"start_time\": 0,\n"
            "    \"end_time\": 15\n"
            "  },\n"
            "  {\n"
            "    \"video_prompt\": \"Example prompt text for the second segment\",\n"
            "    \"video_script\": \"Example script text for the second segment\",\n"
            "    \"start_time\": 15,\n"
            "    \"end_time\": 30\n"
            "  }\n"
            "]\n\n"
            "Video promptlarını ingilizce, video scriptlerini ise Türkçe olarak oluşturun.\n"
            "Lütfen yalnızca geçerli ve temiz bir JSON nesnesi döndürün, ek açıklama veya metin eklemeyin.\n\n"
            "UNUTMA 180 saniye olacak"
            "ÖZET:\n" + summary
        )

        video_script_response = generate_response_with_llm(video_script_prompt)
        try:
            video_script = json.loads(video_script_response)
        except json.JSONDecodeError as e:
            logger.error(f"Video script JSON çözümlenirken hata: {e}")
            video_script = {}

        # Quiz sorularını oluşturmak için yeni prompt:
        quiz_prompt = (
            "Ders kitabından alıp oluşturduğumuz konu özeti ve video scriptini kullanarak, video segmentlerini aşağıdaki gruplara göre "
            "5 şıklı çoktan seçmeli sorular oluşturun:\n"
            "1. İlk 3 segment için bir soru (grup: segment 1–3),\n"
            "2. 4. ile 6. segmentler için bir soru (grup: segment 4–6),\n"
            "3. 7. ile 10. segmentler için bir soru (grup: segment 7–10).\n"
            "Her soru, ilgili segmentlerin toplam başlangıç ve bitiş zamanlarını içermeli. Örnek format:\n"
            "{\n"
            "  \"start_time\": <toplam başlangıç>,\n"
            "  \"end_time\": <toplam bitiş>,\n"
            "  \"question\": \"Soru metni\",\n"
            "  \"answer\": \"Çözüm metni\"\n"
            "}\n\n"
            "ÖZET:"
            f"{summary}\n"
            "Video Script:\n" 
            f"{json.dumps(video_script)}\n"
            "Lütfen yalnızca geçerli ve temiz bir JSON nesnesi döndürün, ek açıklama veya metin eklemeyin."
        )

        quiz_response = generate_response_with_llm(quiz_prompt)
        try:
            quiz_output = json.loads(quiz_response)
        except json.JSONDecodeError as e:
            logger.error(f"Quiz JSON çözümlenirken hata: {e}")
            quiz_output = {}

        return video_script, quiz_output

    else:
        print("İlgili parça bulunamadı. Kullanıcı sorgusu doğrudan işleniyor.")
        response = generate_response_with_llm(user_query)
        print("\nGenerated Response:")
        print(response)
        return None, None