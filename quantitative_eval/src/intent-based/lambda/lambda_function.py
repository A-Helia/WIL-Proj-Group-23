# -*- coding: utf-8 -*-

# This sample demonstrates handling intents from an Alexa skill using the Alexa Skills Kit SDK for Python.
# Please visit https://alexa.design/cookbook for additional examples on implementing slots, dialog management,
# session persistence, api calls, and more.
# This sample is built using the handler classes approach in skill builder.
import logging
import ask_sdk_core.utils as ask_utils

from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.dispatch_components import AbstractExceptionHandler
from ask_sdk_core.handler_input import HandlerInput

from ask_sdk_model import Response

import random

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LaunchRequestHandler(AbstractRequestHandler):
    """Handler for Skill Launch."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool

        return ask_utils.is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        #speak_output = "Hi, I'm Walert, How can I help you?"
        
        response_variations = ["Greetings! I'm Walert, I'm can assist with all your questions about programs at R.M.I.T School of Computing Technologies.",
        "Hi, I'm Walert, DO you have questions about the programs at R.M.I.T School of Computing Technologies.",
        "Good day, I'm Walert, I'm here to help with questions about programs at R.M.I.T School of Computing Technologies.",
        "Hello there! I'm Walert, I can assist with your inquiries about programs at R.M.I.T School of Computing Technologies."]
        
        speak_output = random.choice(response_variations)
        
        ask_variations = ['Feel free to ask if you have any other questions.', 
        "Is there anything else you'd like to inquire about?", 
        "Are there any other inquiries you'd like to make?",
        "If you have any other questions, I'm here to help."]
        
        ask_output = random.choice(ask_variations)
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(ask_output)
                .response
        )


class HelloWorldIntentHandler(AbstractRequestHandler):
    """Handler for Hello World Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("HelloWorldIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speak_output = "Hello World!"

        return (
            handler_input.response_builder
                .speak(speak_output)
                # .ask("add a reprompt if you want to keep the session open for the user to respond")
                .response
        )


class HelpIntentHandler(AbstractRequestHandler):
    """Handler for Help Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("AMAZON.HelpIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        # TODO Ranked order of question.
        
        response_variations = ["You can ask me questions about programs and courses at R.M.I.T's School of Computing Technologies such as Computer Science, Software Engineering, Information Technology and Data Science. For example, you ask can me to tell a little bit about the computer science program."]
        #speak_output = "You can say hello to me! How can I help?"
        
        speak_output = random.choice(response_variations)
        
        ask_output = "Please feel free to ask questions about our courses and programs or you could ask about one of the research groups at R.M.I.T called Seeda."
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )


class CancelOrStopIntentHandler(AbstractRequestHandler):
    """Single handler for Cancel and Stop Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return (ask_utils.is_intent_name("AMAZON.CancelIntent")(handler_input) or
                ask_utils.is_intent_name("AMAZON.StopIntent")(handler_input))

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        logger.info("In CancelOrStopIntentHandler")
        response_variations = ["It's been a pleasure engaging with you. I hope I've successfully resolved any questions you had. Goodbye and Have a great day",
        "Thank you for this interactive session. I hope that I've left no question unanswered. Goodbye for now.",
        "I'm grateful for the opportunity to talk to you. I hope that all your queries have been sufficiently answered. Goodbye and Have a great day"]
        
        should_end_session = True
        
        speak_output = random.choice(response_variations)
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .set_should_end_session(should_end_session)
                .response
        )

class FallbackIntentHandler(AbstractRequestHandler):
    """Single handler for Fallback Intent."""
    def __init__(self):
        self.fallback_count = 0
    
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("AMAZON.FallbackIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        
        self.fallback_count += 1
        if self.fallback_count >= 30:
            
            return handler_input.response_builder.speak("Sorry, I am having technical difficulties right now. Please talk to one of the staff around, while I reboot. Thank you for your patience").set_should_end_session(True).response

        else:
            logger.info("In FallbackIntentHandler")
            
            
            response_variations = ["I'm sorry, I do no have an answer for that question. Please ask the staff around you.",
            "My apologies, I do not have an answer for that question. Please ask the staff around you."]
            
            # "I regret that I didn't catch your question. Would you mind restating it?",
            # "My apologies, your query didn't come across clearly. Could you please repeat it?",
            # "I'm sorry, but I had trouble understanding your question. Could you please reiterate it?"
            
            speak_output = random.choice(response_variations)
            
            reprompt = "What else can i help you with?"

            return handler_input.response_builder.speak(speak_output).ask(reprompt).response

class SessionEndedRequestHandler(AbstractRequestHandler):
    """Handler for Session End."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_request_type("SessionEndedRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response

        # Any cleanup logic goes here.

        return handler_input.response_builder.response


class IntentReflectorHandler(AbstractRequestHandler):
    """The intent reflector is used for interaction model testing and debugging.
    It will simply repeat the intent the user said. You can create custom handlers
    for your intents by defining them above, then also adding them to the request
    handler chain below.
    """
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_request_type("IntentRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        intent_name = ask_utils.get_intent_name(handler_input)
        speak_output = "You just triggered " + intent_name + "."

        return (
            handler_input.response_builder
                .speak(speak_output)
                # .ask("add a reprompt if you want to keep the session open for the user to respond")
                .response
        )


class CatchAllExceptionHandler(AbstractExceptionHandler):
    """Generic error handling to capture any syntax or routing errors. If you receive an error
    stating the request handler chain is not found, you have not implemented a handler for
    the intent being invoked or included it in the skill builder below.
    """
    def can_handle(self, handler_input, exception):
        # type: (HandlerInput, Exception) -> bool
        return True

    def handle(self, handler_input, exception):
        # type: (HandlerInput, Exception) -> Response
        logger.error(exception, exc_info=True)

        speak_output = "Sorry, I had trouble doing what you asked. Please try again."
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )

# # Request interceptor to save the last intent to session attributes
# class SaveLastIntentInterceptor(AbstractRequestInterceptor):
#     def process(self, handler_input):
#         last_intent = ask_utils.get_intent_name(handler_input.request_envelope.request)
#         handler_input.attributes_manager.session_attributes["last_intent"] = last_intent
# # Custom Intent Handler to repeat the last intent
# class RepeatLastIntentHandler(AbstractRequestHandler):
#     def can_handle(self, handler_input):
#         return (ask_utils.is_intent_name("AMAZON.RepeatIntent")(handler_input) and
#                 hasattr(handler_input, "last_intent"))

#     def handle(self, handler_input):
#         if "last_intent" in session_attributes:
#             last_intent = session_attributes["last_intent"]
#             # Dispatch the request to the last executed intent
#             speech_text = "your last intent was " + last_intent

#             return handler_input.response_builder.speak(speech_text).response
#         else:
#             speech_text = "I'm afraid I cannot do that as we just started our conversation. Can you please make a new request?"
#             reprompt_text = "If you are not sure what to ask, you can say the command, HELP ME"
#             return handler_input.response_builder.speak(speech_text).ask(reprompt_text).response

# **************** Custom Intent Handlers below **************************

class ElectivesIntentHandler(AbstractRequestHandler):
    """Handler for Electives Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("Electives")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        
        logger.info("In ElectivesIntentHandler")
        
        response_variations = ["Programs offer various electives like programming, cybersecurity, and more. You can specialise by choosing related electives, or choose broadly. Electives outside of the program require approval.",
        "Each program provides a wide array of electives, enabling specialisation or broad learning. Electives outside the given structure need program manager's approval.",
        "Electives in IT subjects allow specialisation or general learning. Permission is required for electives outside the set program.",
        "Various electives facilitate specialisation in IT sub-disciplines or broad learning. Approval is needed for electives beyond the program structure.",
        "Programs offer diverse electives for specialisation or wider learning. Electives outside the standard program need manager approval."
        ]
        
        speak_output = random.choice(response_variations)
        
        ask_variations = ['Feel free to ask if you have any other questions.', 
        "Is there anything else you'd like to inquire about?", 
        "Are there any other inquiries you'd like to make?",
        "If you have any other questions, I'm here to help."]
        
        ask_output = random.choice(ask_variations)
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(ask_output)
                .response
        )

class DoubleDegreeBusinessIntentHandler(AbstractRequestHandler):
    """Handler for Electives Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("Double_Degree_Business")(handler_input)

    def handle(self, handler_input):
        
        logger.info("In DoubleDegreeBusinessIntentHandler")
        
        response_variations = [
            "While we don't offer a structured double degree, students can choose from numerous cross-disciplinary majors and minors, some with a strong business focus. BIT students have the widest range of options.",
            "There isn't a specific double degree, but we do have a selection of cross-disciplinary majors and minors, with a few emphasising business. BIT students have the greatest choice.",
            "We may not provide a formal double degree, but many business-centric cross-disciplinary majors and minors are available. BIT students have the most diverse options.",
            "Although there's no designated double degree, we present a variety of cross-disciplinary majors and minors, some with a business orientation. BIT students have the most extensive selection.",
            "We don't have a designated double degree, but we offer a multitude of cross-disciplinary majors and minors, some with a significant business component. BIT students have the broadest range to choose from."
            ]
        
        # type: (HandlerInput) -> Response
        speak_output = random.choice(response_variations)
        #speak_output = "Programs offer various electives like programming, cybersecurity, and more. You can specialise by choosing related electives, or choose broadly. Electives outside of the program require approval."
        
        ask_variations = ['Feel free to ask if you have any other questions.', 
        "Is there anything else you'd like to inquire about?", 
        "Are there any other inquiries you'd like to make?",
        "If you have any other questions, I'm here to help."]
        
        ask_output = random.choice(ask_variations)
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(ask_output)
                .response
        )

class DoubleDegreeCSIntentHandler(AbstractRequestHandler):
    """Handler for Electives Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("Double_Degree_CS")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        
        logger.info("In DoubleDegreeCSIntentHandler")
        
        response_variations = ["We offer a single double degree program, namely Computer Science paired with Computer and Network Engineering. This is a 5-year course where two-fifths is dedicated to Computer Science and three-fifths to engineering.",
        "We provide a double degree option in Computer Science and Computer and Network Engineering. This spans over five years, with 40 percent of the time focused on Computer Science and the remaining 60 percent on engineering.",
        "Our institution has a double degree program: Computer Science combined with Computer and Network Engineering. It's a 5-year course, with Computer Science comprising two out of five parts, and engineering making up the rest.",
        "We possess a singular double degree, which is Computer Science and Computer and Network Engineering. It's a five-year program, with a 40-60 split between Computer Science and engineering, respectively.",
        "Our educational offerings include one double degree: Computer Science along with Computer and Network Engineering. This five-year program allocates two-fifths of the curriculum to Computer Science and three-fifths to engineering."
        ]
        
        speak_output = random.choice(response_variations)
        
        ask_variations = ['Feel free to ask if you have any other questions.', 
        "Is there anything else you'd like to inquire about?", 
        "Are there any other inquiries you'd like to make?",
        "If you have any other questions, I'm here to help."]
        
        ask_output = random.choice(ask_variations)
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(ask_output)
                .response
        )

class AccredSEIntentHandler(AbstractRequestHandler):
    """Handler for Electives Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("Accred_SE")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        
        logger.info("In AccredSEIntentHandler")
        
        response_variations = ["R.M.I.T's SE degree, focused on software development, doesn't teach engineering design or advanced math, but includes a discrete maths subject. It meets A.C.M, I triple E, and A.C.S standards.",
        "R.M.I.T's S.E degree, oriented towards software development, lacks engineering design principles or advanced math, but requires one discrete maths course. It's internationally and ACS-accredited.",
        "R.M.I.T's software-centric SE degree excludes engineering design and advanced math, barring one discrete maths subject. It fulfills A.C.M, I triple E, and A.C.S guidelines.",
        "R.M.I.T's software engineering degree, with a software development focus, omits engineering design and advanced math, except for a discrete maths module. It adheres to A.C.M, I triple E, and A.C.S standards.",
        "R.M.I.T's software engineering degree emphasizes software development, not engineering design or advanced math, except for one discrete maths subject. It's compliant with international software engineering industry and ACS standards."]
        
        speak_output = random.choice(response_variations)
        
        ask_variations = ['Feel free to ask if you have any other questions.', 
        "Is there anything else you'd like to inquire about?", 
        "Are there any other inquiries you'd like to make?",
        "If you have any other questions, I'm here to help."]
        
        ask_output = random.choice(ask_variations)
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(ask_output)
                .response
        )

class AccredACSIntentHandler(AbstractRequestHandler):
    """Handler for Electives Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("Accred_ACS")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        response_variations = ["Except the Data Science degree, all other programs are accredited by A.C.S at professional level."]
        
        speak_output = random.choice(response_variations)
        
        ask_variations = ['Feel free to ask if you have any other questions.', 
        "Is there anything else you'd like to inquire about?", 
        "Are there any other inquiries you'd like to make?",
        "If you have any other questions, I'm here to help."]
        
        ask_output = random.choice(ask_variations)
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(ask_output)
                .response
        )

class WorkPlacementIntentHandler(AbstractRequestHandler):
    """Handler for Electives Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("Work_Placement")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        response_variations = ["While not directly, we help students connect with industry contacts and prepare for placements. In worst-case scenarios, we may transfer S.E students to the C.S program.",
        "We don't provide placements directly but assist students with industry connections and placement preparation. S.E students can switch to the C.S program if necessary.",
        "We don't place students directly but offer industry contacts and preparation for placements. If unsuccessful, we transfer S.E students to the C.S program.",
        "Though not directly, we support students with industry networking and readiness for placements. SE students may be moved to the CS program if required.",
        "We don't directly offer placements but facilitate industry connections and preparation. If all fails, S.E students could be transferred to the C.S program."]
        
        speak_output = random.choice(response_variations)
        
        ask_variations = ['Feel free to ask if you have any other questions.', 
        "Is there anything else you'd like to inquire about?", 
        "Are there any other inquiries you'd like to make?",
        "If you have any other questions, I'm here to help."]
        
        ask_output = random.choice(ask_variations)
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(ask_output)
                .response
        )

class InternshipPaidIntentHandler(AbstractRequestHandler):
    """Handler for Electives Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("Internship_Paid")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        response_variations = ["Indeed, according to governmental rules, internships within Australia are paid positions. But bear in mind, if you opt for an internship abroad, the pay regulations of the host country will apply.",
        "Certainly, per the Australian government's regulations, all internships in Australia are paid. However, if you select an internship outside Australia, the remuneration may be subject to the rules of that particular country.",
        "Absolutely, as per government rules, internships within Australia should be paid. But if you decide to intern overseas, the pay guidelines of the host country might come into effect.",
        "Yes, by government mandate, all internships in Australia are paid. But if your internship is in a foreign country, that country's pay regulations would apply.",
        "Indeed, all internships in Australia are paid as per government regulations. However, should you opt for an internship abroad, you would be subject to the regulations of the host country."]
        
        speak_output = random.choice(response_variations)
        
        ask_variations = ['Feel free to ask if you have any other questions.', 
        "Is there anything else you'd like to inquire about?", 
        "Are there any other inquiries you'd like to make?",
        "If you have any other questions, I'm here to help."]
        
        ask_output = random.choice(ask_variations)
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(ask_output)
                .response
        )

class DegreeTypeIntentHandler(AbstractRequestHandler):
    """Handler for Electives Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("Degree_Type")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        response_variations = ["R.M.I.T offers discipline-specific degrees such as Computer Science, Software Engineering, Information Technology, and Data Science. A one-year Honours degree in Computer Science is available after a three-year course. The Deans Scholar degree isn't offered in 2023.",
        "R.M.I.T provides distinct degrees like Computer Science, Software Engineering, I.T, and Data Science. An additional one-year Honours degree in Computer Science is available post three-year course completion. The Deans Scholar degree isn't available in 2023.",
        "R.M.I.T offers specific degrees such as Computer Science, Software Engineering, I.T, and Data Science. Post a three-year course, a one-year Honours degree in Computer Science can be pursued. The Deans Scholar degree isn't offered in 2023.",
        "At R.M.I.T, students can choose from distinct degrees like Computer Science, Software Engineering, IT, and Data Science. A post-three-year-course one-year Honours degree in Computer Science is available. The Deans Scholar degree isn't offered in 2023.",
        "R.M.I.T specializes in degrees like Computer Science, Software Engineering, I.T, and Data Science. An additional one-year Honours degree in Computer Science can be taken after a three-year course. The Deans Scholar degree isn't available in 2023."]
        
        speak_output = random.choice(response_variations)
        
        ask_variations = ['Feel free to ask if you have any other questions.', 
        "Is there anything else you'd like to inquire about?", 
        "Are there any other inquiries you'd like to make?",
        "If you have any other questions, I'm here to help."]
        
        ask_output = random.choice(ask_variations)
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(ask_output)
                .response
        )

class MathReqIntentHandler(AbstractRequestHandler):
    """Handler for Electives Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("Math_Req")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        response_variations = ["In the past, we only admitted students with Maths Methods or Specialist mathematics. Now, we admit students with one of the VCE Maths courses as long as you have minimum of 20 marks in Units 3 and 4."]
        
        speak_output = random.choice(response_variations)
        
        ask_variations = ['Feel free to ask if you have any other questions.', 
        "Is there anything else you'd like to inquire about?", 
        "Are there any other inquiries you'd like to make?",
        "If you have any other questions, I'm here to help."]
        
        ask_output = random.choice(ask_variations)
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(ask_output)
                .response
        )

class ITCSTransferCredIntentHandler(AbstractRequestHandler):
    """Handler for Electives Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("IT_CS_Transfer_Cred")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        response_variations = ["You'll receive credit for shared courses between the two programs, not full credit. We provide a pathway plan, with approximately 75 percent of first-year I.T courses credited.",
        "Full credit isn't granted. However, credit for common courses is given. We offer a pathway plan, where around 75 percent of first-year I.T courses can earn credit.",
        "You won't receive full credit, but credit is given for courses common to both programs. We have a pathway plan, enabling credit for nearly 75 percent of first-year I.T courses.",
        "Full credit isn't possible, but common courses can earn credit. We offer a pathway study plan, with about 75 percent of first-year I.T courses credited.",
        "While full credit isn't provided, you'll get credit for shared courses. Our pathway study plan allows roughly 75 percent of first-year I.T courses to be credited."]
        
        speak_output = random.choice(response_variations)
        
        ask_variations = ['Feel free to ask if you have any other questions.', 
        "Is there anything else you'd like to inquire about?", 
        "Are there any other inquiries you'd like to make?",
        "If you have any other questions, I'm here to help."]
        
        ask_output = random.choice(ask_variations)
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(ask_output)
                .response
        )

class TransferIntentHandler(AbstractRequestHandler):
    """Handler for Electives Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("Transfer")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        response_variations = ["Program transfers depend on the specific programs. Transfers between Software Engineering and Computer Science, or from Software Engineering or Computer Science to Information Technology, are easier during the first two years. However, upward transfers like Information Technology to Computer Science require merit-based assessments.",
        "Transfers between programs vary. It's simpler to switch between Software Engineering and Computer Science or from Software Engineering or Computer Science to Information Technology in the early years. However, transfers from Information Technology to Computer Science require assessments based on merit.",
        "Program transfer ease varies. For instance, switching between Software Engineering and Computer Science or from Software Engineering or Computer Science to Information Technology is easier in the first two years. But, transferring from Information Technology to Computer Science needs certain requirements and is merit-based.",
        "Transfers are easier between some programs, such as Software Engineering and Computer Science or from Software Engineering or Computer Science to Information Technology, especially during the first two years. However, transfers like Information Technology to Computer Science are merit-assessed and require specific criteria.",
        "The ease of program transfers depends. It's easier to transfer between Software Engineering and Computer Science or from Software Engineering or Computer Science to Information Technology in the initial years. Transferring from Information Technology to Computer Science, however, is merit-based and requires certain conditions."]
        
        speak_output = random.choice(response_variations)
        
        ask_variations = ['Feel free to ask if you have any other questions.', 
        "Is there anything else you'd like to inquire about?", 
        "Are there any other inquiries you'd like to make?",
        "If you have any other questions, I'm here to help."]
        
        ask_output = random.choice(ask_variations)
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(ask_output)
                .response
        )

class WILIntentHandler(AbstractRequestHandler):
    """Handler for Electives Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("WIL")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        response_variations = ["A wide variety of Work-Integrated Learning, or W.I.L components, are available for you. These can be both mandatory and optional, depending on your program. If you're studying Software Engineering or any of the professional degrees, a year-long industry placement and a final year project are required. All students need to complete a capstone project in their final year, which may be with an industry partner or simulated as a real industry scenario. For these projects, a dedicated supervisor will work closely with you. Additionally, summer projects and industry events, including ANZ hackathon, A.I.I.A iAwards competition, and various coding or security competitions, are also available each year.",
        "There are numerous Work-Integrated Learning options, some of which are mandatory and others that are optional. As a Software Engineering student or enrolled in any of the professional degrees, you're required to take part in a year of industry placement and a 24-credit-point final year project. It's a requirement for all students to undertake a capstone project in their final year, involving a collaboration with an industry partner or an in-house project mimicking real industry situations. A project supervisor will be assigned to work with you closely. We also have summer projects and other industry events such as the A.N.Z hackathon, A.I.I.A iAwards competition, and various coding or security competitions.",
        "You have a range of Work-Integrated Learning options to choose from, some compulsory and others optional. If you're pursuing Software Engineering or any other professional degree, a year of industry placement and a final year project are mandatory. All students must undertake a capstone project in their final year, where they may work with an industry partner or simulate an industry environment. You will have a project supervisor for guidance. We also offer additional opportunities like summer projects and industry events, such as the A.N.Z hackathon, A.I.I.A iAwards competition, and various coding and security competitions.",
        "R.M.I.T provides an array of Work-Integrated Learning opportunities. Some are essential parts of the programs while others are optional. If you're a Software Engineering student or in any of the professional degrees, a one-year industry placement and a final-year project are required. All RMIT students must carry out a capstone project in their final year, working with an industry partner or in a simulated industry scenario, with a dedicated project supervisor for assistance. Moreover, R.M.I.T offers summer projects and industry events like the A.N.Z hackathon, A.I.I.A iAwards competition, and coding or security competitions each year."]
        
        speak_output = random.choice(response_variations)
        
        ask_variations = ['Feel free to ask if you have any other questions.', 
        "Is there anything else you'd like to inquire about?", 
        "Are there any other inquiries you'd like to make?",
        "If you have any other questions, I'm here to help."]
        
        ask_output = random.choice(ask_variations)
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(ask_output)
                .response
        )

class AssociateBachelorTransferIntentHandler(AbstractRequestHandler):
    """Handler for Electives Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("Associate_Bachelor_Transfer")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        response_variations = ["No, that's not the case. You need to apply when you are nearing the completion of your Associate Degree.",
        "That's incorrect. The application is required when you are close to finishing your Associate Degree.",
        "No, it's not so. You should submit your application when you're about to complete the Associate Degree.",
        "That's not true. You need to apply when you're nearly done with your Associate Degree.",
        "No, that isn't right. The application needs to be made when you are close to the end of your Associate Degree."]
        
        speak_output = random.choice(response_variations)
        
        ask_variations = ['Feel free to ask if you have any other questions.', 
        "Is there anything else you'd like to inquire about?", 
        "Are there any other inquiries you'd like to make?",
        "If you have any other questions, I'm here to help."]
        
        ask_output = random.choice(ask_variations)
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(ask_output)
                .response
        )

class ITFutureAustraliaIntentHandler(AbstractRequestHandler):
    """Handler for Electives Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("IT_Future_Australia")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        response_variations = ["It's a misconception that IT jobs are mostly outsourced. In fact, sectors like startups and innovative technologies are booming locally. Melbourne leads in these areas. Over 90,000 IT jobs were advertised in 2021 alone.",
        "Contrary to popular belief, not all IT jobs are outsourced. Advanced fields like home automation and machine learning are expanding locally, with Melbourne at the forefront. The first half of 2021 saw over 90,000 IT job adverts.",
        "The idea that most IT jobs are outsourced is a myth. Emerging sectors like home automation and IoT aren't outsourced and Melbourne is leading in these areas. There were 91,000 job adverts in the IT sector in early 2021.",
        "Dispelling the myth, not all IT jobs are outsourced. Sectors like startups, innovative tech, and IT-support jobs in SMEs stay local, especially in Melbourne. There were 91,000 IT job adverts in the first half of 2021.",
        "It's inaccurate to think most IT jobs are outsourced. Innovation sectors and IT-support roles in smaller companies often remain local. In 2021's first half, there were 91,000 IT job advertisements."]
        
        speak_output = random.choice(response_variations)
        
        ask_variations = ['Feel free to ask if you have any other questions.', 
        "Is there anything else you'd like to inquire about?", 
        "Are there any other inquiries you'd like to make?",
        "If you have any other questions, I'm here to help."]
        
        ask_output = random.choice(ask_variations)
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(ask_output)
                .response
        )

class ATAR_InsufficientIntentHandler(AbstractRequestHandler):
    """Handler for Electives Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("ATAR_Insufficient")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        response_variations = ["Definitely. Associate degrees usually require lower ATAR scores and provide a feasible route to reach your desired graduation goals.",
        "Indeed. An associate degree typically demands a lower ATAR, making it a suitable pathway to attain your targeted degree.",
        "Without a doubt. Associate degrees, which generally require lower ATAR, can be an effective way to achieve your intended educational outcome.",
        "Certainly. With typically lower ATAR prerequisites, associate degrees can be an ideal route to fulfill your graduation aspirations.",
        "Absolutely. Given their generally lower ATAR requirements, associate degrees could be the perfect path for achieving your desired graduation objectives."]
        
        speak_output = random.choice(response_variations)
        
        ask_variations = ['Feel free to ask if you have any other questions.', 
        "Is there anything else you'd like to inquire about?", 
        "Are there any other inquiries you'd like to make?",
        "If you have any other questions, I'm here to help."]
        
        ask_output = random.choice(ask_variations)
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(ask_output)
                .response
        )

class Job_OpportunitiesIntentHandler(AbstractRequestHandler):
    """Handler for Electives Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("Job_Opportunities")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        response_variations = ["Our students find success in various IT fields, including government, large corporations, small businesses, and startups, with many in Melbourne's thriving startup scene.",
        "From governmental roles to startups, our students secure jobs across the IT spectrum. Some have even launched their own startups, particularly in Melbourne's bustling scene.",
        "Our students excel in multiple IT sectors, from government to startups, including launching their own businesses, especially within Melbourne's vibrant startup environment.",
        "Students from our institution have made their mark across IT sectors including government, IT firms, startups, with a notable presence in Melbourne's burgeoning startup scene.",
        "Spanning government, corporate, and startup sectors, our students have found career success in IT. Some have even spearheaded their own startups in Melbourne's lively entrepreneurial ecosystem."]
        
        speak_output = random.choice(response_variations)
        
        ask_variations = ['Feel free to ask if you have any other questions.', 
        "Is there anything else you'd like to inquire about?", 
        "Are there any other inquiries you'd like to make?",
        "If you have any other questions, I'm here to help."]
        
        ask_output = random.choice(ask_variations)
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(ask_output)
                .response
        )

class JobOpportunitiesGamesIndustryIntentHandler(AbstractRequestHandler):
    """Handler for Electives Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("Job_Opportunities_GamesIndustry")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        response_variations = ["Indeed, the future prospects for careers in the gaming industry, and in a larger context, digital media, are very promising.",
        "Certainly, there is a positive career projection for the games industry, and more broadly, for the digital media sector.",
        "Absolutely, the job outlook for the games industry and, more extensively, the digital media field, is quite optimistic.",
        "Yes, the forecast for career opportunities in the games industry and, in a wider sense, digital media, is very encouraging.",
        "Indeed, the career prospects in the games industry, and more comprehensively, the digital media sector, are highly positive."]
        
        speak_output = random.choice(response_variations)
        
        ask_variations = ['Feel free to ask if you have any other questions.', 
        "Is there anything else you'd like to inquire about?", 
        "Are there any other inquiries you'd like to make?",
        "If you have any other questions, I'm here to help."]
        
        ask_output = random.choice(ask_variations)
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(ask_output)
                .response
        )

class ComparisonBCSBCSProfIntentHandler(AbstractRequestHandler):
    """Handler for Electives Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("Comparison_BCS_BCSProf")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        response_variations = ["The Professional version, introduced for 2023, allows high-performing regular students to transfer for an added internship year, similar to the Software Engineering program, offering a variety of career path choices.",
        "The new Professional version gives top-performing regular program students a chance to add an internship year, aligning with the Software Engineering course, while expanding potential career paths.",
        "The Professional version, launched in 2023, is an opportunity for standout students in the regular program to undertake an extra internship year, akin to Software Engineering, but with broader career options.",
        "Started in 2023, the Professional version enables high-achieving regular program students to gain an internship year, much like Software Engineering students, but with a wider choice of career paths.",
        "The Professional version, that started in 2023, allows exceptional regular program students to incorporate an internship year, similar to Software Engineering, with diverse career possibilities in fields like cyber security, data science, and machine learning."]
        
        speak_output = random.choice(response_variations)
        
        ask_variations = ['Feel free to ask if you have any other questions.', 
        "Is there anything else you'd like to inquire about?", 
        "Are there any other inquiries you'd like to make?",
        "If you have any other questions, I'm here to help."]
        
        ask_output = random.choice(ask_variations)
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(ask_output)
                .response
        )

class ComparisonCSSEIntentHandler(AbstractRequestHandler):
    """Handler for Electives Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("Comparison_CS_SE")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        response_variations = ["A Computer Science degree develops a skill set that span from theoretical and algorithmic foundations to cutting-edge developments in computing. Whereas a Software Engineering degree is centered on the design, development and management of large quality-measured software systems"]
        
        speak_output = random.choice(response_variations)
        
        ask_variations = ['Feel free to ask if you have any other questions.', 
        "Is there anything else you'd like to inquire about?", 
        "Are there any other inquiries you'd like to make?",
        "If you have any other questions, I'm here to help."]
        
        ask_output = random.choice(ask_variations)
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(ask_output)
                .response
        )

class ComparisonCSITIntentHandler(AbstractRequestHandler):
    """Handler for Electives Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("Comparison_CS_IT")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        response_variations = ["CS program covers Computer Science core body of knowledge and will provide a solid foundation in the underlying concepts. It strikes a good balance between the theory and the application of these theories to the application of them into real-life scenarios (in application development). The IT program tip more into the applications with more emphasis on hands-on skills in various aspects of information and communication technology."]
        
        speak_output = random.choice(response_variations)
        
        ask_variations = ['Feel free to ask if you have any other questions.', 
        "Is there anything else you'd like to inquire about?", 
        "Are there any other inquiries you'd like to make?",
        "If you have any other questions, I'm here to help."]
        
        ask_output = random.choice(ask_variations)
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(ask_output)
                .response
        )

class AboutmeIntentHandler(AbstractRequestHandler):
    """Handler for Electives Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("About_me")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        response_variations = ["I am Walert, a conversational assistant developed at R.M.I.T University in Melbourne. I can answer questions regarding the courses and programs at R.M.I.T's School of Computing Technologies."]
        
        speak_output = random.choice(response_variations)
        
        ask_variations = ['Feel free to ask if you have any other questions.', 
        "Is there anything else you'd like to inquire about?", 
        "Are there any other inquiries you'd like to make?",
        "If you have any other questions, I'm here to help."]
        
        ask_output = random.choice(ask_variations)
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(ask_output)
                .response
        )


class CreatorIntentHandler(AbstractRequestHandler):
    """Handler for Electives Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("Creator")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        response_variations = ["I was created by members of the Seeda research group of R.M.I.T University, Melbourne. haha",
        "Members of the Seeda research group at R.M.I.T University, Melbourne, are responsible for my creation. Some of them are around you right now, haha",
        "The brilliant minds at R.M.I.T University's Seeda research group in Melbourne brought me into existence. Some of them are around you right now, haha",
        "I owe my existence to the Seeda research group from R.M.I.T University, Melbourne. Some of them are around you right now, haha",
        "My developers are the esteemed members of the Seeda research group, based in R.M.I.T University, Melbourne. Some of them are around you right now, haha",
        "I originated from the innovative work of the Seeda research group at R.M.I.T University, Melbourne. Some of them are around you right now, haha",
        "The Seeda research group at R.M.I.T University, Melbourne, played a crucial role in my development. Some of them are around you right now, haha",
        "I am a product of the collaborative efforts of the Seeda research group from R.M.I.T University, Melbourne. Some of them are around you right now, haha"]
        
        speak_output = random.choice(response_variations)
        
        ask_variations = ['Feel free to ask if you have any other questions.', 
        "Is there anything else you'd like to inquire about?", 
        "Are there any other inquiries you'd like to make?",
        "If you have any other questions, I'm here to help."]
        
        ask_output = random.choice(ask_variations)
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(ask_output)
                .response
        )

class CIDDAIntentHandler(AbstractRequestHandler):
    """Handler for Electives Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("CIDDA")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        response_variations = ["Seeda stands for Centre for Information Discovery and Data Analytics. Seeda specialises in developing new approaches to find relevant information in massive data collections.",
        "Seeda stands for Centre for Information Discovery and Data Analytics. Specializing in cutting-edge techniques, Seeda excels in discovering relevant insights from massive datasets.",
        "Seeda stands for Centre for Information Discovery and Data Analytics. Seeda stands out for its ability to develop and apply cutting-edge techniques to process vast amounts of data and obtain relevant information."]
        
        speak_output = random.choice(response_variations)
        
        ask_variations = ['Feel free to ask if you have any other questions.', 
        "Is there anything else you'd like to inquire about?", 
        "Are there any other inquiries you'd like to make?",
        "If you have any other questions, I'm here to help."]
        
        ask_output = random.choice(ask_variations)
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(ask_output)
                .response
        )


class BTSIntentHandler(AbstractRequestHandler):
    """Handler for Electives Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("BTS")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        response_variations = ["I use automatic speech recognition and natural language understanding to interpret and respond to your questions related to R.M.I.T, its programs, courses and more.",
        "Using automatic speech recognition and natural language understanding, I respond to your R.M.I.T-related queries, including programs and courses.",
        "I employ automatic speech recognition and natural language understanding to address your questions about R.M.I.T, its programs, courses, and more.",
        "Using automatic speech recognition and natural language understanding, I provide accurate responses to your inquiries related to R.M.I.T, its programs, courses, and more."]
        
        speak_output = random.choice(response_variations)
        
        ask_variations = ['Feel free to ask if you have any other questions.', 
        "Is there anything else you'd like to inquire about?", 
        "Are there any other inquiries you'd like to make?",
        "If you have any other questions, I'm here to help."]
        
        ask_output = random.choice(ask_variations)
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(ask_output)
                .response
        )


class MotivationIntentHandler(AbstractRequestHandler):
    """Handler for Electives Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("Motivation")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        response_variations = ["I was created to provide extensive support to R.M.I.T students, offering quick and easy access to information for an enhanced educational journey. Simplifying your experience, whether you're a new student or seeking R.M.I.T program and staff details."]
        
        speak_output = random.choice(response_variations)
        
        ask_variations = ['Feel free to ask if you have any other questions.', 
        "Is there anything else you'd like to inquire about?", 
        "Are there any other inquiries you'd like to make?",
        "If you have any other questions, I'm here to help."]
        
        ask_output = random.choice(ask_variations)
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(ask_output)
                .response
        )


class AIEnquiryIntentHandler(AbstractRequestHandler):
    """Handler for Electives Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("AI_Enquiry")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        response_variations = ["I am powered by artificial intelligence technology. My creators harnessed the capabilities of A.I to design me as an intelligent and efficient voice assistant. This technology enables me to understand your voice, process your inquiries, and provide tailored responses, making interactions with me more natural and intuitive."]
        
        speak_output = random.choice(response_variations)
        
        ask_variations = ['Feel free to ask if you have any other questions.', 
        "Is there anything else you'd like to inquire about?", 
        "Are there any other inquiries you'd like to make?",
        "If you have any other questions, I'm here to help."]
        
        ask_output = random.choice(ask_variations)
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(ask_output)
                .response
        )


class PersonalityIntentHandler(AbstractRequestHandler):
    """Handler for Electives Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("Personality")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        response_variations = ["While I don't have personal emotions or consciousness, my creators have designed me to be a helpful and friendly companion in your journey at R.M.I.T . Think of me as a knowledgeable guide who is always ready to provide you with accurate information."]
        
        speak_output = random.choice(response_variations)
        
        ask_variations = ['Feel free to ask if you have any other questions.', 
        "Is there anything else you'd like to inquire about?", 
        "Are there any other inquiries you'd like to make?",
        "If you have any other questions, I'm here to help."]
        
        ask_output = random.choice(ask_variations)
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(ask_output)
                .response
        )


class AutoLearnIntentHandler(AbstractRequestHandler):
    """Handler for Electives Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("AutoLearn")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        response_variations = ["While I don't possess the ability to learn and evolve in the same way humans do, my creators regularly update and enhance my knowledge and capabilities. This ensures that I stay up-to-date with the latest information about R.M.I.T, its courses, programs, and any changes that may occur over time."]
        
        speak_output = random.choice(response_variations)
        
        ask_variations = ['Feel free to ask if you have any other questions.', 
        "Is there anything else you'd like to inquire about?", 
        "Are there any other inquiries you'd like to make?",
        "If you have any other questions, I'm here to help."]
        
        ask_output = random.choice(ask_variations)
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(ask_output)
                .response
        )

class CoursesIntentHandler(AbstractRequestHandler):
    """Handler for Electives Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("Courses")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        response_variations = ["Here at R.M.I.T's School of Computing Technologies, we offer courses in Computer Science, Software Engineering, Information Technology, Data Science and many more. Please check with the student connect services for more details."]
        
        speak_output = random.choice(response_variations)
        
        ask_variations = ['Feel free to ask if you have any other questions.', 
        "Is there anything else you'd like to inquire about?", 
        "Are there any other inquiries you'd like to make?",
        "If you have any other questions, I'm here to help."]
        
        ask_output = random.choice(ask_variations)
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(ask_output)
                .response
        )

class ComparisonBachelorAssociateIntentHandler(AbstractRequestHandler):
    """Handler for Electives Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("Comparison_Bachelors_Associate")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        response_variations = ["Associate degrees are normally pathway degrees to a specific bachelor degree. For example, Associate degree in I.T is a pathway to Bachelor of I.T . It roughly covers first two years in the bachelors degree and then you can complete the remainder of the bachelors degree in one year"]
        
        speak_output = random.choice(response_variations)
        
        ask_variations = ['Feel free to ask if you have any other questions.', 
        "Is there anything else you'd like to inquire about?", 
        "Are there any other inquiries you'd like to make?",
        "If you have any other questions, I'm here to help."]
        
        ask_output = random.choice(ask_variations)
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(ask_output)
                .response
        )


class SummaryIntentHandler(AbstractRequestHandler):
    """Handler for Electives Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("Summary")(handler_input)

    def handle(self, handler_input):
        
        slot = ask_utils.request_util.get_slot(handler_input, "program_name")
        
        
        # Get resolutionsPerAuthority
        resolutions_per_authority = handler_input.request_envelope.request.intent.slots["program_name"].resolutions.resolutions_per_authority
        
        if resolutions_per_authority:
            resolved_slot_type = resolutions_per_authority[0].values[0].value.name
            
            if resolved_slot_type == 'ds':
                speak_output = 'The Data science program is focused on the data problems, so it would have a narrower computing focus but more mathematical and statistical in nature.'
            elif resolved_slot_type =='cs':
                speak_output = 'The computer science program is most technically focused. Would give the deepest technical knowledge. Technical leaders will do this program.'
            elif resolved_slot_type == 'it':
                speak_output = 'The information technology program is most flexible, it allows combining information technology with other areas of study.'
            elif resolved_slot_type == 'se':
                speak_output = 'The software engineering program is similar to the computer science program, but with  a project management focus. It is for project leaders rather than technical leaders.'
            else:
                speak_output = 'I am afraid I have no knowledge about this program or course. Could you check with the Student connect ?'
            
        # # type: (HandlerInput) -> Response
        # response_variations = ["Here at R.M.I.T's School of Computing Technologies, we offer courses in Computer Science, Software Engineering, Information Technology, Data Science and many more. Please check with the student connect services for more details."]
        
        # speak_output = random.choice(response_variations)
        
        ask_variations = ['Feel free to ask if you have any other questions.', 
        "Is there anything else you'd like to inquire about?", 
        "Are there any other inquiries you'd like to make?",
        "If you have any other questions, I'm here to help."]
        
        ask_output = random.choice(ask_variations)
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(ask_output)
                .response
        )


# The SkillBuilder object acts as the entry point for your skill, routing all request and response
# payloads to the handlers above. Make sure any new handlers or interceptors you've
# defined are included below. The order matters - they're processed top to bottom.


sb = SkillBuilder()

# sb.add_request_handler(RepeatLastIntentHandler())

# sb.add_global_request_interceptor(SaveLastIntentInterceptor())

sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(HelloWorldIntentHandler())
sb.add_request_handler(HelpIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_request_handler(FallbackIntentHandler())
sb.add_request_handler(SessionEndedRequestHandler())

sb.add_request_handler(ElectivesIntentHandler())
sb.add_request_handler(DoubleDegreeBusinessIntentHandler())
sb.add_request_handler(DoubleDegreeCSIntentHandler())
sb.add_request_handler(AccredSEIntentHandler())
sb.add_request_handler(AccredACSIntentHandler())
sb.add_request_handler(WorkPlacementIntentHandler())
sb.add_request_handler(InternshipPaidIntentHandler())
sb.add_request_handler(DegreeTypeIntentHandler())
sb.add_request_handler(MathReqIntentHandler())
sb.add_request_handler(ITCSTransferCredIntentHandler())
sb.add_request_handler(ITFutureAustraliaIntentHandler())
sb.add_request_handler(TransferIntentHandler())
sb.add_request_handler(WILIntentHandler())
sb.add_request_handler(AssociateBachelorTransferIntentHandler())
sb.add_request_handler(ATAR_InsufficientIntentHandler())
sb.add_request_handler(Job_OpportunitiesIntentHandler())
sb.add_request_handler(JobOpportunitiesGamesIndustryIntentHandler())
sb.add_request_handler(ComparisonBCSBCSProfIntentHandler())
sb.add_request_handler(ComparisonCSSEIntentHandler())
sb.add_request_handler(ComparisonCSITIntentHandler())
sb.add_request_handler(AboutmeIntentHandler())
sb.add_request_handler(CreatorIntentHandler())
sb.add_request_handler(CIDDAIntentHandler())
sb.add_request_handler(BTSIntentHandler())
sb.add_request_handler(MotivationIntentHandler())
sb.add_request_handler(AIEnquiryIntentHandler())
sb.add_request_handler(PersonalityIntentHandler())
sb.add_request_handler(AutoLearnIntentHandler())
sb.add_request_handler(SummaryIntentHandler())
sb.add_request_handler(ComparisonBachelorAssociateIntentHandler())

sb.add_request_handler(IntentReflectorHandler()) # make sure IntentReflectorHandler is last so it doesn't override your custom intent handlers

sb.add_exception_handler(CatchAllExceptionHandler())

lambda_handler = sb.lambda_handler()