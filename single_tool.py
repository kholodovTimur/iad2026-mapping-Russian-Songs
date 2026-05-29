from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import START, END, StateGraph
from geopy.location import Location
from huggingface_hub import hf_hub_download

from llama_cpp import Llama

from pydantic import BaseModel, Field
from typing import Any, Literal, TypedDict

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

import geopandas as gpd
import numpy as np
import pandas as pd
import wandb
import time
import json
import os

from tqdm.auto import tqdm

# Структура взята из документации: ссылка!
from pydantic import PrivateAttr

class LlamaLLM(BaseChatModel):
    model_path: str
    llm: Llama

    @property
    def _llm_type(self) -> str:
        return "llama-cpp-python"
    
    def __init__(self, model_path: str, **kwargs: Any):
        main_kwargs = {k: v for k, v in kwargs.items() if k in {'n_ctx', 'n_batch', 'n_threads', 'verbose', 'use_mmap'}}
        llm = Llama(model_path=model_path, n_gpu_layers=-1, use_mmap=False, **main_kwargs)
        super().__init__(model_path=model_path, llm=llm)
        
        
    def _convert_messages(self, messages: list) -> list[dict]:
        dictionary = {'ai':'assistant', 'human': 'user', 'system':'system'}
        return [{'role': dictionary[i.type], 'content': i.content} for i in messages]
        

    def _generate(self, messages: list, **kwargs) -> ChatResult:
        
        messages = self._convert_messages(messages)
        response = self.llm.create_chat_completion(
            messages=messages, **kwargs
        )
        
        return  ChatResult(generations = [ChatGeneration(message = AIMessage(content = response['choices'][0]['message']['content']))])

    def with_structured_output(self, schema, include_raw: bool = False):
        
        def _generate_strucured(messages: list, **kwargs):
            
            messages = self._convert_messages(messages)

            try:
                response = self.llm.create_chat_completion(
                    messages=messages,
                    response_format={'type': 'json_object', 'schema': schema.model_json_schema()},
                    **kwargs
                )
            except Exception as e:
                if include_raw:
                    return {'raw': None, 'parsed': None, 'gen_error': e, 'error_phase': 'generation'}
                raise
            
            response = response['choices'][0]['message']['content']
            
            parsed_response = None
            try:
                parsed_response = schema.model_validate_json(response)
            except Exception as e:
                if include_raw:
                    return {'raw': AIMessage(content=response), 'parsed': None, 'error': e, 'error_phase': 'parsing'}
                raise
                
            if include_raw:
                return {
                    'raw': AIMessage(content=response),
                    'parsed': parsed_response,
                    'error': None,
                    'error_phase': None
                }

            return parsed_response
        
        return RunnableLambda(_generate_strucured)
    
class Place(BaseModel):
    toponym: str = Field(
        description=jsonprompts['ToponymDescr']
    )
    normal: str = Field(
        description=jsonprompts['NormalizDescr']
    )
    type: Literal[
        "улица", "метро", "район",
        "город", "регион", "округ", "страна", "природа", "другое"
    ] = Field(
        description=(
            jsonprompts['TypeTopDescr']
        )
    )

class SongInfo(BaseModel):
    places: list[Place] = Field(
        description="Все именованные географические объекты, "
                    "найденные в тексте песни. Пустой список — если не найдено."
    )

class AddressMatch(BaseModel):
    match: bool = Field(
        description="True если адрес соответствует топониму в контексте песни"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Уверенность от 0.0 до 1.0"
    )
    
class Song(TypedDict):
    song_text: str
    toponymns: SongInfo
    locations: list[Location]
    confident_topomymns: list[str] = []
    score: str
    recognitiontime: float
    geotime: float

class SongToponymGeoRecognition:
    
    def __init__(self, recognition_model_repo_id: str = "unsloth/Qwen3.6-35B-A3B-GGUF", recognition_model_name: str = "Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf",
                   geo_model_repo_id: str = "unsloth/Qwen3.6-35B-A3B-GGUF", geo_model_name: str = "Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf",
                   recognition_mt: int = 512, recognition_t: float = 0.7, recog_ctx: int = 4096, geo_mt: int = 512, geo_t: float = 0.1, geo_ctx: int = 4096, **kwargs):
        self.recognition_model_repo_id = recognition_model_repo_id
        self.recognition_model_name = recognition_model_name
        self.geo_model_repo_id = geo_model_repo_id
        self.geo_model_name = geo_model_name
        self.recognition_mt = recognition_mt
        self.recognition_t = recognition_t
        self.recog_ctx = recog_ctx
        self.geo_mt = geo_mt
        self.geo_t = geo_t
        self.geo_ctx = geo_ctx
        self.model = self.graph_instance(**kwargs)
    
    def get_model(self, model_repo_id: str, model_name: str, mt: int, t: float, ctx: int, **kwargs):
        
        if 'gemini' in model_name.lower():
            
            if not os.environ.get('GOOGLE_API_KEY', '').strip():
                os.environ['GOOGLE_API_KEY'] = input('Input your Google API key: ')
            
            using_model = ChatGoogleGenerativeAI(model=model_name,
                                                temperature=t,
                                                max_tokens=mt,
                                                max_retries=2,
                                                **kwargs)
        else:
            if not os.environ.get('HF_TOKEN', '').strip():
                os.environ['HF_TOKEN'] = input('Input your HuggingFace API key: ')
                
            model_path = hf_hub_download(
                    repo_id=model_repo_id,
                    filename=model_name
                )
            
            using_model = LlamaLLM(model_path=model_path,
                n_ctx=ctx, **kwargs)
    
        return using_model
    
    def get_toponymns(self, model, state: Song):
        st_t = time.time()
        
        prompt = jsonprompts['MainToponymPrompt']
            
        coder = model.with_structured_output(SongInfo)
        state['toponymns'] = coder.invoke([SystemMessage('Отвечай без размышлений. /no_think'),
                                    HumanMessage(f'{prompt}\nВот текст песни: {state['song_text']}')])  
        f_t = time.time()
        full_t = f_t-st_t
        print(f'Распознавание топонимов заняло {full_t}')
        state['recognitiontime'] = full_t
        return state

    def get_context(self, song_text: str, toponym: str):
        try:
            idx = song_text.find(toponym)
            if idx == -1:
                return song_text
    
            if '\n' in song_text and len(song_text.split('\n')) > 10:
                lines = song_text.split('\n')

                char_count = 0
                target_line = 0
                for i, line in enumerate(lines):
                    if (char_count + len(line)) >= idx:
                        target_line = i
                        break
                    char_count += len(line) - 1
    
                start = max(0, target_line - 5)
                end = min(len(lines), target_line + 6)
                return '\n'.join(lines[start:end])
    
            else:
                start = max(0, idx - 150)
                end = min(len(song_text), idx + len(toponym) + 150)
                return song_text[start:end]
    
        except Exception:
            return song_text
    
    def get_geo(self, model, state: Song):
        geolocator = Nominatim(user_agent="iad_project")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        locations = []
        confident_topomymns = []
        st_t = time.time()
        for toponym in [i for i in state['toponymns'].model_dump()['places']]:
            toponym_loc = geocode(toponym['normal'], geometry = 'geojson', exactly_one = True)

            if toponym_loc != None:

                coder = model.with_structured_output(AddressMatch)
        
        
                prompt = jsonprompts['MainGeoPrompt']

                context = self.get_context(state['song_text'], toponym['toponym'])
                
                ans = coder.invoke([SystemMessage('Отвечай без размышлений. /no_think'),
                            HumanMessage(prompt.format(context=context,
                                                        toponym=toponym['normal'],
                                                        address=toponym_loc.address))], max_tokens=self.geo_mt, temperature=self.geo_t)  # Здесь температура и максимальный размер токенов в случае если они отличаются от их размера для геокодирования
                if ans.match == True:
                    locations.append(toponym_loc)
                    confident_topomymns.append(toponym['toponym'])
            else:
                continue
            f_t = time.time()
            full_t = f_t-st_t
        print(f'Распознование гео заняло: {full_t}')
            
        state['locations'] = locations
        state['confident_topomymns'] = confident_topomymns
        state['geotime'] = full_t
        return state

    def should_continue(self, state):
        finded = len(state['toponymns'].places)
        if finded == 0:
            state['confident_topomymns'] = []
            return END
        else:
            return 'get_geo'

    def graph_instance(self, **kwargs):

        if (self.recognition_model_repo_id == self.geo_model_repo_id) and (self.recognition_model_name == self.geo_model_name):
            equal_model = self.get_model(self.recognition_model_repo_id, self.recognition_model_name, self.recognition_mt, self.recognition_t, self.recog_ctx, **kwargs)

            recog_model = equal_model
            geo_model = equal_model
        else:
            recog_model = self.get_model(self.recognition_model_repo_id, self.recognition_model_name, self.recognition_mt, self.recognition_t, self.recog_ctx, **kwargs)
            geo_model = self.get_model(self.geo_model_repo_id, self.geo_model_name, self.geo_mt, self.geo_t, self.geo_ctx, **kwargs)

        graph_builder = StateGraph(Song)

        graph_builder.add_node("get_toponyms", lambda state: self.get_toponymns(recog_model, state))
        graph_builder.add_node("get_geo", lambda state: self.get_geo(geo_model, state))

        graph_builder.add_edge(START, "get_toponyms")

        graph_builder.add_conditional_edges(
            "get_toponyms",
            self.should_continue,
            ['get_geo', END])

        graph_builder.add_edge("get_geo", END)

        graph = graph_builder.compile()

        return graph
    
    def proceed_one_track(self, song: str):
        """
        Позволяет обработать один текст песни при помощи инициализированной модели.
        """
        return self.model.invoke({'song_text': song})
        
    def proceed_data(self, data, run_name, project_name = 'iadProject', notes=''):
        """
        Позволяет обработать датасет из песен при помощи инициализированной модели.
        """

        def to_dict(cell):
            if pd.isna(cell) == True:
                return []
            else:
                cell = cell.split(', ')
                cell = [{'toponym': i.split(':')[0], 'type': i.split(':')[:-1]} for i in cell]
                return cell

        df = data.copy()
                
        df['Топонимы'] = df['Топонимы'].apply(lambda x: to_dict(x))
                
        for col in ['predicted', 'true_toponyms', 'true_types', 'predicted_toponyms', 'predicted_types', 'true_predicted', 'unpredicted', 'overpredicted', 'TP', 'FP', 'FN']:
            df[col] = None
            df[col] = df[col].astype(object)

        with tqdm(df.iterrows(), total = len(df)) as pbar:

            hyperparameters = {
                'model': run_name or self.recognition_model_name,
                'temp_recog': self.recognition_t,
                'temp_geo': self.geo_t,
                'prompt': '-',
                'context_window_recog': self.recog_ctx,
                'context_window_geo': self.geo_ctx,
                'max_tokens_recog': self.recognition_mt,
                'max_tokens_geo': self.geo_mt
            }
            
            with wandb.init(project=project_name, config=hyperparameters, name = run_name or self.recognition_model_name, notes=notes) as run:
                nan = 0
                recog_time = 0
                geo_time = 0
                for index, data in pbar:
                    try:
                        run.define_metric("technical/total_time", summary="mean")
                        run.define_metric("technical/recognition_time", summary="mean")
                        run.define_metric("technical/geo_time", summary="mean")
                        start = time.time()
                        
                        p = self.model.invoke({'song_text': data['lyrics']})
                        
                        if len(p['toponymns'].places) == 0:
                            p = []
                            recog_time = nan
                            geo_time = nan
                        else:
                            recog_time = p['recognitiontime']
                            geo_time = p['geotime']
                            p = p['confident_topomymns']
                            
                            
                        df.at[index, 'predicted'] = p
                            
                        total_time = time.time() - start
                        
                        df.at[index, 'true_toponyms'] = [i['toponym'] for i in data['Топонимы']]
                        df.at[index, 'predicted_toponyms'] = df.at[index, 'predicted']
                        
                        df.at[index, 'true_predicted'] = set(df.loc[index, 'true_toponyms']) & set(df.loc[index, 'predicted_toponyms'])
                        print(f'Верные топонимы: {set(df.loc[index, 'true_toponyms'])}, Предсказанные: {set(df.loc[index, 'predicted_toponyms'])}')
                        df.at[index, 'unpredicted'] = set(df.loc[index, 'true_toponyms']) - set(df.loc[index, 'predicted_toponyms'])
                        df.at[index, 'overpredicted'] = set(df.loc[index, 'predicted_toponyms']) - set(df.loc[index, 'true_toponyms'])
                        
                        df.at[index, 'TP'] = len(df.at[index, 'true_predicted'])
                        df.at[index, 'FN'] = len(df.at[index, 'unpredicted'])
                        df.at[index, 'FP'] = len(df.at[index, 'overpredicted'])

                    except Exception as e:
                        total_time = time.time() - start
                        
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
                        tqdm.write(f"Error: {e}")

                    pbar.set_postfix({
                        'TP': df.at[index, 'TP'],
                        'FN': df.at[index, 'FN'],
                        'FP': df.at[index, 'FP']
                        
                    })

                    text_accuracy = len(df[(df['FN'] == 0) & (df['FP'] == 0)].loc[:index]) / len(df.loc[:index])

                    if (truepos_falseneg_sum := (df['TP'].sum() + df['FN'].sum())) > 0:
                        recall = df['TP'].sum() / truepos_falseneg_sum
                    else:
                        recall = 0

                    if (truepos_falsepos_sum := (df['TP'].sum() + df['FP'].sum())) > 0:
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
                            'technical/total_time': total_time,
                            'technical/recognition_time': recog_time,
                            'technical/geo_time': geo_time
                    })

                
                artifact = wandb.Artifact("words", type="dataset")
                words = wandb.Table(dataframe = df[['unpredicted','overpredicted']])
                artifact.add(words, "unpredicted_overpredicted")
                wandb.log_artifact(artifact)
                
                

        return df

