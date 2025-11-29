# GPT4 things ########################################################################
from openai import OpenAI

class OpenAIHandler:
    def __init__(self, api_key, model_id):
        self.conversation_log = []
        self._api_key = api_key
        self.model_id = model_id
        self.client = OpenAI(api_key=self._api_key)

    def get_model_response(self):
        # Update conversation log with new utterance from user
        #new_user_turn_formatted = {"role": "user", "content": new_user_turn}
        #self.conversation_log.append(new_user_turn_formatted)

        # Complete the conversation with OPENAI Model of choice (defined in constructor)
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=self.conversation_log
        )

        return response

    def add_model_turn(self, response):
        response = {"role": response.choices[0].message.role, "content": response.choices[0].message.content.strip()}
        self.conversation_log.append(response)

    def add_user_turn(self, user_turn_utterance):
        user_turn = {"role": "user", "content": user_turn_utterance}
        self.conversation_log.append(user_turn)

    def remove_turns(self, i_begin, i_end=None):
        if i_end:
            del self.conversation_log[i_begin:i_end]
        else:
            del self.conversation_log[i_begin]
