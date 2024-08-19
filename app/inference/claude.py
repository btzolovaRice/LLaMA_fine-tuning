from anthropic import AnthropicBedrock
from .plan_generator import PlanGenerator, GeneratedPlan, ModelClass

# Ideally could probably use RAG to retrieve relevant examples
UTTERANCE_PLAN_PROMPT_TEMPLATE = """
You are an intelligent assistant specialized in generating structured LISP-like plans based on user instructions (utterances). Your task is to convert natural language instructions into well-defined plans. Each plan should reflect the intent expressed in the utterance and follow a specific structure.

Here are some examples:

1. **Utterance**: "create chess tournament on monday in chicago"
   **Plan**: "( Yield :output ( CreateCommitEventWrapper :event ( CreatePreflightEventWrapper :constraint ( Constraint[Event] :location ( ?= # ( LocationKeyphrase \" chicago \" ) ) :start ( Constraint[DateTime] :date ( ?= ( NextDOW :dow # ( DayOfWeek \" MONDAY \" ) ) ) ) :subject ( ?= # ( String \" chess tournament \" ) ) ) ) ) )"

2. **Utterance**: "Who is james manager ?"
   **Plan**: "( Yield :output ( FindManager :recipient ( Execute :intension ( refer ( extensionConstraint ( RecipientWithNameLike :constraint ( Constraint[Recipient] ) :name # ( PersonName \" james \" ) ) ) ) ) ) "

3. **Utterance**: "Who is my bosses manager ?"
   **Plan**: "( Yield :output ( FindManager :recipient ( FindManager :recipient ( toRecipient ( CurrentUser ) ) ) ) )"

4. **Utterance**: "I have a meeting with Jim tomorrow at 2 pm that will last an hour in Jim 's office"
   **Plan**: "( Yield :output ( CreateCommitEventWrapper :event ( CreatePreflightEventWrapper :constraint ( Constraint[Event] :attendees ( AttendeeListHasRecipient :recipient ( Execute :intension ( refer ( extensionConstraint ( RecipientWithNameLike :constraint ( Constraint[Recipient] ) :name # ( PersonName \" Jim \" ) ) ) ) ) ) :duration ( ?= ( toHours # ( Number 1 ) ) ) :location ( ?= # ( LocationKeyphrase \" Jim's office \" ) ) :start ( ?= ( DateAtTimeWithDefaults :date ( Tomorrow ) :time ( NumberPM :number # ( Number 2 ) ) ) ) ) ) ) )"

5. **Utterance**: "Can you add another Meeting and Greeting with Sasha , Quinn , and Katia for the last Friday in March at 1 : 00 PM"
   **Plan**: "( Yield :output ( CreateCommitEventWrapper :event ( CreatePreflightEventWrapper :constraint ( Constraint[Event] :attendees ( andConstraint ( andConstraint ( AttendeeListHasRecipient :recipient ( Execute :intension ( refer ( extensionConstraint ( RecipientWithNameLike :constraint ( Constraint[Recipient] ) :name # ( PersonName \" Sasha \" ) ) ) ) ) ) ( AttendeeListHasRecipient :recipient ( Execute :intension ( refer ( extensionConstraint ( RecipientWithNameLike :constraint ( Constraint[Recipient] ) :name # ( PersonName \" Quinn \" ) ) ) ) ) ) ) ( AttendeeListHasRecipient :recipient ( Execute :intension ( refer ( extensionConstraint ( RecipientWithNameLike :constraint ( Constraint[Recipient] ) :name # ( PersonName \" Katia \" ) ) ) ) ) ) ) :start ( ?= ( DateAtTimeWithDefaults :date ( DowOfWeekNew :dow # ( DayOfWeek \" FRIDAY \" ) :week ( NumberWeekFromEndOfMonth :month # ( Month \" MARCH \" ) :number # ( Number 1 ) ) ) :time ( NumberPM :number # ( Number 1 ) ) ) ) :subject ( ?= # ( String \" Meeting and Greeting \" ) ) ) ) ) )"

6. **Utterance**: "I would like to create an appointment for getting my cat neutered for the 13 th at 3 pm ."
   **Plan**: "( Yield :output ( CreateCommitEventWrapper :event ( CreatePreflightEventWrapper :constraint ( Constraint[Event] :start ( ?= ( DateAtTimeWithDefaults :date ( nextDayOfMonth ( Today ) # ( Number 13 ) ) :time ( NumberPM :number # ( Number 3 ) ) ) ) :subject ( ?= # ( String \" getting my cat neutered \" ) ) ) ) ) )"

7. **Utterance**: "Can you schedule an appointment for dinner with Bailey and Jenna for Tuesday sometime between 5 and 8 depending on their schedule ?"
   **Plan**: "( Yield :output ( CreateCommitEventWrapper :event ( CreatePreflightEventWrapper :constraint ( Constraint[Event] :attendees ( andConstraint ( AttendeeListHasRecipient :recipient ( Execute :intension ( refer ( extensionConstraint ( RecipientWithNameLike :constraint ( Constraint[Recipient] ) :name # ( PersonName \" Bailey \" ) ) ) ) ) ) ( AttendeeListHasRecipient :recipient ( Execute :intension ( refer ( extensionConstraint ( RecipientWithNameLike :constraint ( Constraint[Recipient] ) :name # ( PersonName \" Jenna \" ) ) ) ) ) ) ) :end ( ?= ( TimeAfterDateTime :dateTime ( DateAtTimeWithDefaults :date ( NextDOW :dow # ( DayOfWeek \" TUESDAY \" ) ) :time ( NumberPM :number # ( Number 5 ) ) ) :time ( NumberPM :number # ( Number 8 ) ) ) ) :start ( ?= ( DateAtTimeWithDefaults :date ( NextDOW :dow # ( DayOfWeek \" TUESDAY \" ) ) :time ( NumberPM :number # ( Number 5 ) ) ) ) :subject ( ?= # ( String \" dinner \" ) ) ) ) ) )"

8. **Utterance**: "Is Abby on my team ?"
   **Plan**: "( Yield :output ( PersonOnTeam :person ( PersonFromRecipient :recipient ( Execute :intension ( refer ( extensionConstraint ( RecipientWithNameLike :constraint ( Constraint[Recipient] ) :name # ( PersonName \" Abby \" ) ) ) ) ) ) :team ( FindTeamOf :recipient ( toRecipient ( CurrentUser ) ) ) ) )"

   
Now, given the utterance, generate a structured plan that accurately reflects the user's request.

Only give me the generated plan, do not include any other text.
"""

class ClaudePlanGenerator(PlanGenerator):
    def __init__(self, bedrock_client: AnthropicBedrock):
        super().__init__()
        self._client = bedrock_client

    def generate_plan(self, utterance):
        message = self._client.messages.create(
            model="anthropic.claude-3-5-sonnet-20240620-v1:0",
            max_tokens=512,
            system=UTTERANCE_PLAN_PROMPT_TEMPLATE,
            messages=[{"role": "user", "content": utterance}]
        )

        return GeneratedPlan(plan=message.content[0].text, confidence=None, model_class=ModelClass.LARGE)
