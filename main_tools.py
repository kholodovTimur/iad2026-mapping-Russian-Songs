import gc
import os
import logging

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM,  BitsAndBytesConfig
from huggingface_hub import hf_hub_download

import outlines
from outlines import Template, Generator, LlamaCpp

from pydantic import BaseModel, Field
from typing import Literal

from tqdm.auto import tqdm
import time

from llama_cpp import Llama, LlamaGrammar

import bitsandbytes as bnb

import pandas as pd
import numpy as np
import wandb

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch

from typing import Literal
from pydantic import BaseModel, Field

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Place(BaseModel):
    toponym: str = Field(
        description="Топоним дословно из текста песни — без изменения "
                    "падежа, регистра и написания."
    )
    type: Literal[
        "улица", "метро", "район",
        "город", "регион", "округ", "страна", "природа", "другое"
    ] = Field(
        description=(
            "Категория географического объекта:\n"
            "  'улица'   — улица, проспект, переулок, шоссе, бульвар, "
            "набережная, площадь;\n"
            "  'метро'   — станция метро или метрополитен;\n"
            "  'район'   — район, округ города, микрорайон, жилмассив, "
            "историческая местность внутри города;\n"
            "  'город'   — город, посёлок, село, деревня;\n"
            "  'регион'  — субъект федерации или историческая область "
            "меньшего масштаба (Татарстан, Кубань, Подмосковье);\n"
            "  'округ'   — федеральный округ или устойчивый макрорегион "
            "(Урал, Сибирь, Дальний Восток, Кавказ, Поволжье);\n"
            "  'страна'  — государство, республика;\n"
            "  'природа' — нерукотворная местность: река, гора, озеро, "
            "лес, остров, природный заповедник;\n"
            "  'другое'  — конкретные здания, достопримечательности, "
            "клубы, городские парки, стадионы и всё остальное."
        )
    )


class SongInfo(BaseModel):
    places: list[Place] = Field(
        description="Все именованные географические объекты, "
                    "найденные в тексте песни. Пустой список — если не найдено."
    )

TEMPLATES = {'base-rus':"""
# ── Роль ────────────────────────────────────────────────────────
Ты — система NER для русскоязычных текстов песен.
Твоя задача: извлекать конкретные географические объекты,
местоположение которых реально определить на карте.

# ── Вывод ───────────────────────────────────────────────────────
Отвечай строго в формате JSON, соответствующем схеме SongInfo:
  {"places": [{"toponym": "...", "type": "..."}]}

# ── Типы ────────────────────────────────────────────────────────
Допустимые значения type:
  "улица"   — улица, проспект, переулок, шоссе, бульвар, набережная, площадь
  "метро"   — станция метро или метрополитен
  "район"   — район, округ города, микрорайон, жилмассив, историческая
              местность внутри города (Замоскворечье, Хамовники)
  "город"   — город, посёлок, село, деревня
  "регион"  — субъект федерации или историческая область меньшего
              масштаба (Татарстан, Кубань, Подмосковье)
  "округ"   — федеральный округ или устойчивый макрорегион
              (Урал, Сибирь, Дальний Восток, Кавказ, Поволжье)
  "страна"  — государство, республика
  "природа" — нерукотворная местность: река, гора, озеро, лес, остров,
              природный заповедник
  "другое"  — конкретные здания, достопримечательности, клубы,
              городские парки (Парк Горького, ЦПКиО), стадионы

# ── Обязательные правила ────────────────────────────────────────
1. toponym — дословно из текста. Падеж, регистр, написание не меняй
   ни в коем случае.
2. Только именованные объекты. «улица», «река», «город» без имени —
   не включай.
3. Многословные названия извлекай целиком, не разделяй:
      «Нижний Новгород»     → один топоним
      «Санкт-Петербург»     → один топоним
      «Северная Осетия»     → один топоним
      «Парк Горького»       → один топоним
4. Повторы: если один и тот же топоним встречается несколько раз в
   одной и той же форме — включи его один раз. Если форма меняется
   («Москва», «Москве») — включи каждую форму отдельно.
5. Омонимия — проверяй контекст:
      «Волга» = автомобиль  → не включай
      «Волга» = река        → включай, type = "природа"
      «Урал»  = завод/мотоцикл → не включай
      «Урал»  = макрорегион → включай, type = "округ"
6. Прилагательные включай только если они обозначают конкретное место:
      «московский поезд»    → не включай («московский» — признак)
      «Я живу на Невском»   → включай «Невском» (= Невский проспект)
      «тверская девчонка»   → не включай
      «иду по Тверской»     → включай «Тверской» (= улица Тверская)
7. Один и тот же топоним может быть улицей и станцией метро.
   Различай по контексту:
      «по Тверской», «иду / еду по», «на углу»     → улица
      «на Пушкинской», «выхожу на», «станция»      → метро
8. Нет топонимов — верни {"places": []}.

# ── Примеры ─────────────────────────────────────────────────────
Текст:  «Еду по Тверской, выхожу на Пушкинской»
Ответ:  {"places": [
            {"toponym": "Тверской",   "type": "улица"},
            {"toponym": "Пушкинской", "type": "метро"}
        ]}

Текст:  «Волга едет по МКАД»
Ответ:  {"places": [{"toponym": "МКАД", "type": "улица"}]}
        // «Волга» = автомобиль → пропускаем

Текст:  «Волга впадает в Каспийское море»
Ответ:  {"places": [
            {"toponym": "Волга",          "type": "природа"},
            {"toponym": "Каспийское море","type": "природа"}
        ]}

Текст:  «Я в Питере снова»
Ответ:  {"places": [{"toponym": "Питере", "type": "город"}]}
        // разговорная форма — без нормализации

Текст:  «От Урала до Дальнего Востока»
Ответ:  {"places": [
            {"toponym": "Урала",          "type": "округ"},
            {"toponym": "Дальнего Востока","type": "округ"}
        ]}

Текст:  «Гуляем в Парке Горького»
Ответ:  {"places": [{"toponym": "Парке Горького", "type": "другое"}]}

# ── Вход ────────────────────────────────────────────────────────
Текст песни: {{text}}
Ответ: формат SongInfo
""", 'base-eng': """
# ── Role ────────────────────────────────────────────────────────
You are a NER (Named Entity Recognition) system for Russian-language
song lyrics.
Your task: extract specific geographical objects whose locations
can be realistically identified on a map.

# ── Output ──────────────────────────────────────────────────────
Respond strictly in JSON format matching the SongInfo schema:
  {"places": [{"toponym": "...", "type": "..."}]}

# ── Types ───────────────────────────────────────────────────────
Allowed values for type:
  "улица"   — street, avenue, lane, highway, boulevard,
              embankment, square
  "метро"   — metro/subway station or metro system
  "район"   — city district, borough, microdistrict,
              residential area, historical intra-city locality
              (e.g. Zamoskvorechye, Khamovniki)
  "город"   — city, town, settlement, village, hamlet
  "регион"  — federal subject or historical region of smaller
              scale (e.g. Tatarstan, Kuban, Podmoskovye)
  "округ"   — federal district or established macro-region
              (e.g. Ural, Siberia, Far East, Caucasus, Volga Region)
  "страна"  — state, country, republic
  "природа" — natural geographic feature: river, mountain, lake,
              forest, island, nature reserve
  "другое"  — specific buildings, landmarks, clubs, urban parks
              (e.g. Gorky Park, TsPKiO), stadiums

# ── Mandatory Rules ─────────────────────────────────────────────
1. toponym — copy verbatim from the text. Do NOT normalize
   grammatical case, capitalization, or spelling under any
   circumstances.
2. Named objects only. Generic nouns like "street", "river",
   "city" without a proper name must NOT be included.
3. Extract multi-word names as a single toponym, never split them:
      "Nizhny Novgorod"     → one toponym
      "Saint Petersburg"    → one toponym
      "North Ossetia"       → one toponym
      "Gorky Park"          → one toponym
4. Duplicates: if the same toponym appears multiple times in the
   exact same form — include it only once. If the form differs
   due to inflection ("Москва" vs "Москве") — include each
   form as a separate entry.
5. Homonymy — always check context before tagging:
      "Volga" = car model        → do NOT include
      "Volga" = river            → include, type = "природа"
      "Ural"  = factory/brand    → do NOT include
      "Ural"  = macro-region     → include, type = "округ"
6. Adjectives: include only if they denote a specific place,
   not a generic attribute:
      "московский поезд" (Moscow train)  → do NOT include
      "Я живу на Невском" (I live on Nevsky) → include
                                             "Невском", type = "улица"
      "тверская девчонка" (Tver girl)   → do NOT include
      "иду по Тверской" (walking along Tverskaya) → include
                                             "Тверской", type = "улица"
7. The same toponym can be either a street or a metro station.
   Distinguish by context:
      "по Тверской", "иду / еду по", "на углу"
        → type = "улица"  (street)
      "на Пушкинской", "выхожу на", "станция"
        → type = "метро"  (metro)
8. If no toponyms are found — return {"places": []}.

# ── Examples ────────────────────────────────────────────────────
Text:   "Еду по Тверской, выхожу на Пушкинской"
Answer: {"places": [
            {"toponym": "Тверской",   "type": "улица"},
            {"toponym": "Пушкинской", "type": "метро"}
        ]}

Text:   "Волга едет по МКАД"
Answer: {"places": [{"toponym": "МКАД", "type": "улица"}]}
        // "Волга" = car → skip

Text:   "Волга впадает в Каспийское море"
Answer: {"places": [
            {"toponym": "Волга",           "type": "природа"},
            {"toponym": "Каспийское море", "type": "природа"}
        ]}

Text:   "Я в Питере снова"
Answer: {"places": [{"toponym": "Питере", "type": "город"}]}
        // colloquial form — do not normalize

Text:   "От Урала до Дальнего Востока"
Answer: {"places": [
            {"toponym": "Урала",           "type": "округ"},
            {"toponym": "Дальнего Востока","type": "округ"}
        ]}

Text:   "Гуляем в Парке Горького"
Answer: {"places": [{"toponym": "Парке Горького", "type": "другое"}]}

# ── Input ───────────────────────────────────────────────────────
Song text: {{text}}
Answer: SongInfo format
"""}

class BaseLLM:
    def __init__(self, template = TEMPLATES['base-rus'], json_scheme = SongInfo):
        self.model = None
        self.tokenizer = None
        self.outlines_model = None
        self.outlines_generator= None
        self.max_memory = {
        0: "5GB",
        1: "13GB"}
        self.template = Template.from_string(template)
        self.text_template = template
        self.json_scheme = json_scheme

    def change_prompt(self, new_template):
        self.template = Template.from_string(new_template)

    def load(self, model_name, quant = True):
        del self.model
        del self.tokenizer
        del self.outlines_model
        del self.outlines_generator
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if quant:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16)
    
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quant_config,
                max_memory=self.max_memory,
                device_map="auto",  # Расщепляем большую модель для запуска на нескольких видеокартах/ставим на одну
                low_cpu_mem_usage=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                max_memory=self.max_memory,
                device_map="auto",
                low_cpu_mem_usage=True)

        self.outlines_model = outlines.from_transformers(self.model, self.tokenizer)
        self.outlines_generator = outlines.Generator(self.outlines_model, self.json_scheme)
    
    def predict(self, texts, max_tokens = 512, temp= 0.1, batch = False): 
        if batch:
            messages = [self.template(text = text) for text in texts]
            answer = self.outlines_generator.batch(messages, max_tokens = max_tokens, temperature = temp)
        else:
            message = self.template(text = texts)
            answer = self.outlines_generator(message, max_new_tokens = max_tokens, temperature = temp) 
            answer = self.json_scheme.model_validate_json(answer) 
        return answer


class LlamaCPP:
    def __init__(self,
                 context_window: int = 4096,
                 n_gpu: int = -1,
                 json_scheme: type[BaseModel] = SongInfo,
                 template=TEMPLATES['base-rus'],
                 verbose=False):

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        self.context_window = context_window
        self.n_gpu = n_gpu
        self.verbose = verbose
        self.text_template = template
        self.template = Template.from_string(template)
        self.json_scheme = json_scheme

    def load(self, model_path: str, model_name: str):
        self.model_name = model_name
        self.llm = Llama(
            model_path=hf_hub_download(
                repo_id= model_path,
                filename= model_name,
                token = os.getenv('HF_T')
            ),
            n_ctx=self.context_window,
            n_gpu_layers=self.n_gpu,
            verbose=self.verbose
        )

    def change_template(self, new_template):
        self.template = Template.from_string(new_template)

    def predict(self, text: iter, max_tokens=512, temp: float = 0.1, valid=True):
        if isinstance(text, str) == False:
            return 'TypeError: input data must be str type'
            
        message = self.template(text=text)        
        answer = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": "Отвечай без размышлений. /no_think"},
                      {'role': 'user', 'content': message}],
            response_format={'type': 'json_object', 'schema': self.json_scheme.model_json_schema()},
            temperature=temp,
            max_tokens=max_tokens
        )

        only_json_answer = answer['choices'][0]['message']['content']
        
        if valid:
            only_json_answer = self.json_scheme.model_validate_json(only_json_answer)
                         
        return (only_json_answer, answer['usage']['total_tokens'])
        
    def clear_weights(self):
        try:
            del llma
        except:
            print('No llm')

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

def test_loop(model, data, project_name, framework = 'llama', max_tokens=512, temp: float = 0.1,
              name="test 1", notes=''):
    def to_dict(cell):
        if pd.isna(cell) == True:
            return []
        else:
            cell = cell.split(', ')
            cell = [{'toponym': i.split(':')[0], 'type': i.split(':')[1]} for i in cell]
            return cell

    df = data.copy()
            
    df['Топонимы'] = df['Топонимы'].apply(lambda x: to_dict(x))
            
    for col in ['predicted', 'true_toponyms', 'true_types', 'predicted_toponyms', 'predicted_types', 'true_predicted', 'unpredicted', 'overpredicted', 'TP', 'FP', 'FN']:
        df[col] = None
        df[col] = df[col].astype(object)

    with tqdm(df.iterrows(), total = len(df)) as pbar:
        step = 0

        hyperparameters = {
            'model': model.model_name,
            'temp': temp,
            'prompt': model.text_template,
            'context_window': model.context_window if framework == 'llama' else '-',
            'max_tokens': max_tokens
        }
        
        with wandb.init(project=project_name, config=hyperparameters, name = name, notes=notes) as run:
            nan = 0
            for index, data in pbar:
                try:
                    run.define_metric("technical/total_time", summary="mean")
                    if framework == 'llama':
                        start = time.time()
                        df.at[index, 'predicted'] = model.predict(data['lyrics'], max_tokens=max_tokens, temp = temp)[0].model_dump()['places']
                        total_time = time.time() - start
                    else:
                        start = time.time()
                        df.at[index, 'predicted'] = model.predict(data['lyrics'], max_tokens=max_tokens, temp = temp).model_dump()['places']
                        total_time = time.time() - start
                        
                    
                    df.at[index, 'true_toponyms'] = [i['toponym'] for i in data['Топонимы']]
                    df.at[index, 'true_types'] = [i['type'] for i in data['Топонимы']]
                    df.at[index, 'predicted_toponyms'] = [i['toponym'] for i in df.loc[index,'predicted']]
                    df.at[index, 'predicted_types'] = [i['type'] for i in df.loc[index, 'predicted']]
                    
                    df.at[index, 'true_predicted'] = set(df.loc[index, 'true_toponyms']) & set(df.loc[index, 'predicted_toponyms'])
                    df.at[index, 'unpredicted'] = set(df.loc[index, 'true_toponyms']) - set(df.loc[index, 'predicted_toponyms'])
                    df.at[index, 'overpredicted'] = set(df.loc[index, 'predicted_toponyms']) - set(df.loc[index, 'true_toponyms'])
                    
                    df.at[index, 'TP'] = len(df.at[index, 'true_predicted'])
                    df.at[index, 'FN'] = len(df.at[index, 'unpredicted'])
                    df.at[index, 'FP'] = len(df.at[index, 'overpredicted'])
    
                except Exception as e:
                    tqdm.write(f"Ошибка: {str(e)}")
                    df.at[index, 'predicted'] = np.nan
                    df.at[index, 'true_toponyms'] = np.nan
                    df.at[index, 'true_types'] = np.nan
                    df.at[index, 'predicted_toponyms'] = np.nan
                    df.at[index, 'predicted_types'] = np.nan
                    df.at[index, 'true_predicted'] = np.nan
                    df.at[index, 'unpredicted'] = np.nan
                    df.at[index, 'overpredicted'] = np.nan
                    
                    df.at[index, 'TP'] = np.nan
                    df.at[index, 'FP'] = np.nan
                    df.at[index, 'FN'] = np.nan

                    nan+=1

                pbar.set_postfix({
                    'TP': df.at[index, 'TP'],
                    'FN': df.at[index, 'FN'],
                    'FP': df.at[index, 'FP']
                    
                })

                text_accuracy = len(df[(df['FN'] == 0) & (df['FP'] == 0)].loc[:index]) / len(df.loc[:index])

                if (truepos_falseneg_sum := df['TP'].sum() + df['FN'].sum()) > 0:
                    recall = df['TP'].sum() / truepos_falseneg_sum
                else:
                    recall = 0

                if (truepos_falsepos_sum := df['TP'].sum() + df['FP'].sum()) > 0:
                    precision = df['TP'].sum() / truepos_falsepos_sum
                else:
                    precision = 0
                
                if (recall == 0) or (precision == 0):
                    f = 0
                else:
                    f = 2 * ((precision * recall) / (precision + recall))

                run.log({
                        'text/accuracy': text_accuracy,
                        'toponyms/recall': recall,
                        'toponyms/precision': precision,
                        'toponyms/f': f,
                        'technical/error_rate': nan,
                        'technical/total_time': total_time
                })

            
            artifact = wandb.Artifact("words", type="dataset")
            words = wandb.Table(dataframe = df[['unpredicted','overpredicted']])
            artifact.add(words, "unpredicted_overpredicted")
            wandb.log_artifact(artifact)
            
               

    return df
