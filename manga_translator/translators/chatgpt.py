import re
import openai
import asyncio
import time

from .common import CommonTranslator, MissingAPIKeyException
from .keys import OPENAI_API_KEY, OPENAI_HTTP_PROXY

class ChatGPTTranslator(CommonTranslator):
    _LANGUAGE_CODE_MAP = {
        'CHS': 'Simplified Chinese',
        'CHT': 'Traditional Chinese',
        'CSY': 'Czech',
        'NLD': 'Dutch',
        'ENG': 'English',
        'FRA': 'French',
        'DEU': 'German',
        'HUN': 'Hungarian',
        'ITA': 'Italian',
        'JPN': 'Japanese',
        'KOR': 'Korean',
        'PLK': 'Polish',
        'PTB': 'Portuguese',
        'ROM': 'Romanian',
        'RUS': 'Russian',
        'ESP': 'Spanish',
        'TRK': 'Turkish',
        'UKR': 'Ukrainian',
        'VIN': 'Vietnamese',
    }

    def __init__(self):
        super().__init__()
        openai.api_key = openai.api_key or OPENAI_API_KEY
        if not openai.api_key:
            raise MissingAPIKeyException('Please set the OPENAI_API_KEY environment variable before using the chatgpt translator.')
        if OPENAI_HTTP_PROXY:
            proxies = {
                "http": "http://%s" % OPENAI_HTTP_PROXY,
                "https": "http://%s" % OPENAI_HTTP_PROXY
            }
            openai.proxy = proxies

    async def _translate(self, from_lang, to_lang, queries):
        prompt = f"""Please help me translate the following text from a manga into {to_lang}:
You must follow the format below for your reply.

The content you need to translate will start with "Text" followed by a number. The text to be translated will be on the next line.

For example:
Text 1:
あら… サンタさん 見つかったのですね

The reply must correspond to the source. For example, the translation result for Text 1 should start with "Translation 1:", followed by the content on the next line.
For example:
Translation 1:
哦，聖誕老人啊，你被找到了啊

Here is an example:

----------The source I provided to you--------
Text 1:
あら… サンタさん 見つかったのですね

Text 2:
ご心配 おかけしました！


----------The source I provided to you ends---------

---------Your reply starts----------
Translation 1:
哦，聖誕老人啊，你被找到了啊

Translation 2:
讓你操心了，真不好意思！

----------Your reply ends-----------

The instructions are over. Please translate the following content:
      """
        for i, query in enumerate(queries):
            prompt += f'\nText {i+1}:\n{query}\n'
        self.logger.debug(prompt)

        request_task = asyncio.create_task(self._request_translation(prompt))
        started = time.time()
        attempts = 0
        while not request_task.done():
            await asyncio.sleep(0.1)
            if time.time() - started > 60:
                if attempts >= 3:
                    raise Exception('API servers did not respond quickly enough.')
                self.logger.info(f'Restarting request due to timeout. Attempt: {attempts+1}')
                request_task.cancel()
                request_task = asyncio.create_task(self._request_translation(prompt))
                started = time.time()
                attempts += 1
        response = await request_task

        self.logger.debug(response)
        translations = re.split(r'Translation \d+:\n', response)
        translations = [t.strip() for t in translations if t.strip()]
        self.logger.debug(translations)
        return translations

    async def _request_translation(self, prompt: str) -> str:
        messages = [{"role": "system", "content": "You are a professional translator who will follow the required format for translation."}]
        messages.append({"role": "user", "content": prompt})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # The name of the OpenAI chatbot model to use
            messages=messages,
            temperature=0.7
        )
        for choice in response.choices:
            if "text" in choice:
                return choice.text

        # If no response with text is found, return the first response's content (which may be empty)
        return response.choices[0].message.content
