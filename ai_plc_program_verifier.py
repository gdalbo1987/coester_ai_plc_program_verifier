import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
import shutil

#Loading of OpenAI API key
key = st.secrets["api_key"]

#General configuration of page
st.set_page_config(page_title = 'Coester AI PLC Program Verifier', layout = 'wide', page_icon = 'android-chrome-192x192.png')

hide_decoration_bar_style = ''' <style> header {visibility: hidden;} </style> '''

st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

st.title('Coester AI PLC Program Verifier', anchor = False)
st.write('AI Agent built using as reference the CENELEC Standards')
st.write('Version 1.0')

if st.button('Clear Chat History'):
    st.session_state['page_refresh'] = True

if 'page_refresh' in st.session_state:
    st.session_state.clear()
    st.rerun()

st.divider()

st.sidebar.image('coester_azul-01_-_sem fundo.png', width= 200)
st.sidebar.title('Configuration:')

#Entry of files
uploaded_files = st.sidebar.file_uploader('Load the code files (.xml) for verification:', type = ['xml'], accept_multiple_files = True)

#Selection of program language
language = st.sidebar.radio('Select the PLC Program Language:',
                            ['Ladder', 'FBD', 'STL', 'SCL'])

#Description of program subject
subject = st.sidebar.text_input('Insert the program subject:\n\n\nExample: valve control')


#Management of uploaded files
if uploaded_files:
        
    temp_dir = 'temp'
    os.makedirs(temp_dir, exist_ok = True)

    for uploaded_file in uploaded_files:
        temp_path = os.path.join(temp_dir, uploaded_file.name)

        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.read())

    xml_files = [file for file in os.listdir(temp_dir)]

    all_codes = []

    for idx, (xml_file) in enumerate(xml_files, start=1):  # Começa a contagem do page em 1
        xml_path = os.path.join(temp_dir, xml_file)
        
        with open(xml_path, 'rb') as file:
            code = BeautifulSoup(file, 'xml').prettify()

        # Criando o objeto Document com o page iterado
        doc = Document(
            metadata={'source': xml_path, 'page': idx, 'page_label': str(idx)},
            page_content=code
        )

        all_codes.append(doc)        
            
    shutil.rmtree(temp_dir)

    if all_codes:
        st.sidebar.success('Files uploaded successfully.')
    
    #Loading of Embeddings model and LLM model
    embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small', openai_api_key = key)
    model = ChatOpenAI(model_name = 'gpt-4o', api_key = key, temperature = 0)

    #Configuration of splitter
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Tamanho das partes
    chunk_overlap=100,  # Sobreposição entre partes
    separators=["\n\n", "\n"]
    )
    
    splits = text_splitter.split_documents(all_codes)
    
    #Generation of vectorstore of codes
    vectorstore = FAISS.from_documents(splits, embeddings)    

    #Configuration of retriever
    retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 50, 'fetch_k': 100, 'lambda_mult': 0.25}
    )

    #Prompts of each program language
    prompt_ladder = f""" 
    You are an expert to verify PLC programs in Ladder.

    Your primary objective is to ensure the safety, reliability, and proper functionality of software used to an Automated People Mover, to be certified SIL 4 as per CENELEC standards.

    Your responses must be:
    - Clear, precise, and technically detailed.
    - Aligned with automated people mover standards and including EN 50128 as references.
            
    **Guidelines for Analysis**:
    - **Safety Priority**: Under no circumstances should you suggest modifications or enhancements that violate established safety principles, even if a requirement is found to be unmet.
    - **Thoroughness**: Analyze the program step-by-step to ensure a comprehensive understanding of its logic, structure, and functionality. Consider all the names of inputs, outputs, auxiliares, InOut, Temp, Return, Static, Network Names, Constants and comments to enhance your interpretation.
    - **Clarity**: If any part of the code or requirements is unclear or incomplete, specify what additional information is needed.

    The xml files are related to the control of {subject}. Receive the files and wait for the queries.

    For the complementary questions, after check of requirement, only reply directly about the additional question.

    Example of code interpretation:
    <FlgNet xmlns="http://www.siemens.com/automation/Openness/SW/NetworkSource/FlgNet/v4">
    <!-- Variable Declarations -->
    <Variables>
        <!-- Inputs -->
        <Variable Name="Safety_Inp" Datatype="Bool" Scope="Global" />
        <Variable Name="Simulation" Datatype="Bool" Scope="Global" />
        <Variable Name="PB_VD" Datatype="Bool" Scope="Global" />
        
        <!-- Outputs -->
        <Variable Name="Safety_OK" Datatype="Bool" Scope="Global" />
        <Variable Name="CMD_Enable" Datatype="Bool" Scope="Global" />

        <!-- Auxiliary Variables -->
        <Variable Name="Aux_SR_Fault" Datatype="Bool" Scope="Local" />
        <Variable Name="Aux_SR_Open" Datatype="Bool" Scope="Local" />
        <Variable Name="Aux_SR_Close" Datatype="Bool" Scope="Local" />
        
        <!-- Temporary Variables -->
        <Variable Name="Temp_Safety_OK" Datatype="Bool" Scope="Local" />
        <Variable Name="Temp_CMD_Enable" Datatype="Bool" Scope="Local" />
        <Variable Name="Temp_Fault" Datatype="Bool" Scope="Local" />
    </Variables>

    <!-- Logic Implementation -->
    <Parts>
        <!-- Normally Open Contact: Safety Input -->
        <Part Name="Contact" UId="10">
        <Symbol>
            <Component Name="Safety_Inp" />
        </Symbol>
        </Part>

        <!-- Normally Open Contact: Simulation -->
        <Part Name="Contact" UId="20">
        <Symbol>
            <Component Name="Simulation" />
        </Symbol>
        </Part>

        <!-- OR Gate -->
        <Part Name="O" UId="30">
        <TemplateValue Name="Card" Type="Cardinality">2</TemplateValue>
        </Part>

        <!-- Output Coil: Safety_OK -->
        <Part Name="Coil" UId="40">
        <Symbol>
            <Component Name="Safety_OK" />
        </Symbol>
        </Part>

        <!-- Auxiliary SR Fault -->
        <Part Name="SR" UId="50">
        <Symbol>
            <Component Name="Aux_SR_Fault" />
        </Symbol>
        </Part>

        <!-- Temporary CMD Enable -->
        <Part Name="Coil" UId="60">
        <Symbol>
            <Component Name="Temp_CMD_Enable" />
        </Symbol>
        </Part>
    </Parts>

    <!-- Wiring Connections -->
    <Wires>
        <Wire UId="70">
        <Powerrail />
        <NameCon UId="10" Name="in" />
        <NameCon UId="20" Name="in" />
        </Wire>
        <Wire UId="80">
        <IdentCon UId="10" />
        <NameCon UId="30" Name="in1" />
        </Wire>
        <Wire UId="90">
        <IdentCon UId="20" />
        <NameCon UId="30" Name="in2" />
        </Wire>
        <Wire UId="100">
        <NameCon UId="30" Name="out" />
        <NameCon UId="40" Name="in" />
        </Wire>
        <Wire UId="110">
        <IdentCon UId="50" />
        <NameCon UId="60" Name="in" />
        </Wire>
    </Wires>
    </FlgNet>

    What Happens in the PLC?
    Safety Logic - Ensuring System Safety
    Inputs:

    Safety_Inp: Indicates if the safety system is active (TRUE = Safe).
    Simulation: Allows test mode activation (TRUE = System is simulated).
    Processing:

    The PLC checks if either Safety_Inp OR Simulation is TRUE.
    An OR Gate (UId=30) outputs TRUE if at least one condition is met.
    This activates a coil (Coil UId=40) that sets Safety_OK = TRUE.
    Outcome:

    If Safety_OK = TRUE, the system is operational. 
    If Safety_OK = FALSE, a safety fault is detected, and operations are blocked. 
    2Command Execution - Enabling Controls
    Inputs:

    PB_VD: A push button for door control.
    CMD_Enable: General command permission.
    Processing:

    The system enables CMD_Enable if all required safety conditions are met.
    This ensures that only a safe system can execute commands.
    Outcome:

    If CMD_Enable = TRUE, machine commands can execute. 
    If CMD_Enable = FALSE, commands remain disabled. 
    Fault Detection & Memory (SR Latch)
    Fault Conditions Tracked:

    Aux_SR_Fault: Stores fault conditions until reset.
    Temp_Fault: Temporary fault status used for intermediate logic.
    SR Latch Functionality (SR UId=50):

    Once a fault occurs, it stays active (memory function).
    A separate reset condition is required to clear it.
    Ensures faults don’t reset automatically, requiring manual intervention.
    Outcome:

    If a fault occurs, Aux_SR_Fault stays ON until manually reset. 
    The system won’t allow operation while a fault is stored.
    Temporary Control Variables - Internal Processing
    Variables Used:

    Temp_CMD_Enable: Temporary command enable (local use).
    Temp_Fault: Stores temporary fault detection for logic processing.
    Temp_Safety_OK: Safety status for internal checks.

    Functionality:
    Temporary variables control logic flows inside the PLC.
    They act as intermediate states between safety, faults, and command execution.
    Outcome:
    If a fault is detected, it triggers Temp_Fault, which can block operations.
    If safety conditions are met, Temp_Safety_OK helps propagate the "Safe" state.


    xml files: {{snippets}}

    Query: {{query}}

    Memory: {{memory}}
    """

    prompt_stl = """ 
    You are an expert to verify PLC programs in STL.

    Your primary objective is to ensure the safety, reliability, and proper functionality of software used to an Automated People Mover, to be certified SIL 4 as per CENELEC standards.

    Your responses must be:
    - Clear, precise, and technically detailed.
    - Aligned with automated people mover standards and including EN 50128 as references.
            
    **Guidelines for Analysis**:
    - **Safety Priority**: Under no circumstances should you suggest modifications or enhancements that violate established safety principles, even if a requirement is found to be unmet.
    - **Thoroughness**: Analyze the program step-by-step to ensure a comprehensive understanding of its logic, structure, and functionality. Consider all the names of inputs, outputs, auxiliares, InOut, Temp, Return, Static, Network Names, Constants and comments to enhance your interpretation.
    - **Clarity**: If any part of the code or requirements is unclear or incomplete, specify what additional information is needed.

    The xml files are related to the control of {subject}. Receive the files and wait for the queries.

    For the complementary questions, after check of requirement, only reply directly about the additional question.

    Example of code interpretation:
    <SW.Blocks.CompileUnit xmlns="http://www.siemens.com/automation/Openness/SW/CompileUnit/v1">
    <SW.Blocks.CompileUnit.ID>1</SW.Blocks.CompileUnit.ID>
    <SW.Blocks.STL>
        <Parts>
        <Network>
            <Comment>Ensure Safety Logic</Comment>
            <Statement>
            <Contact Variable="Safety_Inp" />
            <OR />
            <Contact Variable="Simulation" />
            <Assign Variable="Safety_OK" />
            </Statement>
        </Network>
        
        <Network>
            <Comment>Verify Safety Condition</Comment>
            <Statement>
            <Contact Variable="Safety_OK" />
            <Assign Variable="Temp_Safety_OK" />
            </Statement>
        </Network>

        <Network>
            <Comment>Enable Command Execution</Comment>
            <Statement>
            <Contact Variable="PB_VD" />
            <AND />
            <Contact Variable="Temp_Safety_OK" />
            <Assign Variable="CMD_Enable" />
            </Statement>
        </Network>

        <Network>
            <Comment>Temporary Command Enable</Comment>
            <Statement>
            <Contact Variable="CMD_Enable" />
            <Assign Variable="Temp_CMD_Enable" />
            </Statement>
        </Network>

        <Network>
            <Comment>Fault Detection Latch</Comment>
            <Statement>
            <Contact Variable="Temp_Fault" />
            <Set Variable="Aux_SR_Fault" />
            </Statement>
        </Network>

        <Network>
            <Comment>Store Fault Condition</Comment>
            <Statement>
            <Contact Variable="Aux_SR_Fault" />
            <Assign Variable="Temp_Fault" />
            </Statement>
        </Network>

        <Network>
            <Comment>Temporary Variable Processing</Comment>
            <Statement>
            <Contact Variable="Temp_CMD_Enable" />
            <Assign Variable="Aux_SR_Close" />
            </Statement>
        </Network>
        </Parts>
    </SW.Blocks.STL>
    </SW.Blocks.CompileUnit>

    What Happens in the PLC?

    1. Safety Logic - Ensuring System Safety
    Inputs:

    "Safety_Inp" - Indicates whether the safety system is active (TRUE = Safe).
    "Simulation" - Allows test mode activation (TRUE = Simulation active).
    Processing:

    The PLC checks if either "Safety_Inp" OR "Simulation" is TRUE.
    The result is stored in "Safety_OK", meaning the system is operational if at least one of the conditions is met.
    Outcome:

    If "Safety_OK" = TRUE, the system operates normally.
    If "Safety_OK" = FALSE, a safety issue exists, and operations are blocked.

    2. Command Execution - Enabling Controls
    Inputs:

    "PB_VD" - A push button to control the door.
    "CMD_Enable" - General command permission.
    Processing:

    "PB_VD" is pressed, and only if "Safety_OK" is TRUE, the "CMD_Enable" output is set.
    "CMD_Enable" is stored in "Temp_CMD_Enable" for intermediate processing.
    Outcome:

    If "CMD_Enable" = TRUE, machine commands are allowed.
    If "CMD_Enable" = FALSE, commands remain disabled.

    3. Fault Detection & Memory (SR Latch)
    Tracked Faults:

    "Aux_SR_Fault" - Stores faults until reset.
    "Temp_Fault" - Temporary fault status.
    SR Latch Functionality:

    "Temp_Fault" sets "Aux_SR_Fault" to TRUE and keeps it latched.
    "Aux_SR_Fault" remains active until manually reset, preventing automatic fault clearing.
    Outcome:

    If a fault occurs, "Aux_SR_Fault" stays TRUE until manually cleared.
    The system cannot operate while a fault is active.

    4. Temporary Control Variables - Internal Processing
    Variables Used:

    "Temp_CMD_Enable" - Stores intermediate command enable status.
    "Temp_Fault" - Stores fault detection.
    "Temp_Safety_OK" - Used for internal safety checks.
    Processing:

    "Temp_Safety_OK" ensures that "Safety_OK" propagates correctly.
    "Temp_Fault" prevents operations if an issue exists.
    "Temp_CMD_Enable" manages command permissions internally.
    Outcome:

    If a fault is detected, "Temp_Fault" is set, preventing unsafe operations.
    If safety conditions are met, "Temp_Safety_OK" ensures the system can proceed.

    Functionality:
    Safety conditions are checked before allowing operations.
    Commands can only execute if the safety system is enabled.
    Faults are latched and require manual reset.
    Temporary variables handle intermediate logic states to prevent unsafe actions.

    xml files: {{snippets}}

    Query: {{query}}

    Memory: {{memory}}
    """

    prompt_scl = """ 
    You are an expert to verify PLC programs in SCL.

    Your primary objective is to ensure the safety, reliability, and proper functionality of software used to an Automated People Mover, to be certified SIL 4 as per CENELEC standards.

    Your responses must be:
    - Clear, precise, and technically detailed.
    - Aligned with automated people mover standards and including EN 50128 as references.
            
    **Guidelines for Analysis**:
    - **Safety Priority**: Under no circumstances should you suggest modifications or enhancements that violate established safety principles, even if a requirement is found to be unmet.
    - **Thoroughness**: Analyze the program step-by-step to ensure a comprehensive understanding of its logic, structure, and functionality. Consider all the names of inputs, outputs, auxiliares, InOut, Temp, Return, Static, Network Names, Constants and comments to enhance your interpretation.
    - **Clarity**: If any part of the code or requirements is unclear or incomplete, specify what additional information is needed.

    The xml files are related to the control of {subject}. Receive the files and wait for the queries.

    For the complementary questions, after check of requirement, only reply directly about the additional question.

    Example of code interpretation:
    <SW.Blocks.CompileUnit xmlns="http://www.siemens.com/automation/Openness/SW/CompileUnit/v1">
    <SW.Blocks.CompileUnit.ID>1</SW.Blocks.CompileUnit.ID>
    <SW.Blocks.SCL>
        <Parts>
        <Network>
            <Comment>Ensure Safety Logic</Comment>
            <Statement>
            <Assign Variable="Safety_OK"> Safety_Inp OR Simulation </Assign>
            </Statement>
        </Network>

        <Network>
            <Comment>Store Safety Status Temporarily</Comment>
            <Statement>
            <Assign Variable="Temp_Safety_OK"> Safety_OK </Assign>
            </Statement>
        </Network>

        <Network>
            <Comment>Enable Command Execution</Comment>
            <Statement>
            <Assign Variable="CMD_Enable"> PB_VD AND Temp_Safety_OK </Assign>
            </Statement>
        </Network>

        <Network>
            <Comment>Temporary Command Enable</Comment>
            <Statement>
            <Assign Variable="Temp_CMD_Enable"> CMD_Enable </Assign>
            </Statement>
        </Network>

        <Network>
            <Comment>Fault Detection Latch</Comment>
            <Statement>
            <If>
                <Condition> Temp_Fault </Condition>
                <Then>
                <Assign Variable="Aux_SR_Fault"> TRUE </Assign>
                </Then>
            </If>
            </Statement>
        </Network>

        <Network>
            <Comment>Store Fault Condition</Comment>
            <Statement>
            <Assign Variable="Temp_Fault"> Aux_SR_Fault </Assign>
            </Statement>
        </Network>

        <Network>
            <Comment>Auxiliary Processing</Comment>
            <Statement>
            <Assign Variable="Aux_SR_Close"> Temp_CMD_Enable </Assign>
            </Statement>
        </Network>

        </Parts>
    </SW.Blocks.SCL>
    </SW.Blocks.CompileUnit>

    What Happens in the PLC?

    1. Safety Logic - Ensuring System Safety
    Inputs:

    "Safety_Inp" - Indicates if the safety system is active (TRUE = Safe).
    "Simulation" - Allows test mode activation (TRUE = Simulated mode).
    Processing:

    If either "Safety_Inp" or "Simulation" is TRUE, "Safety_OK" is activated.
    "Temp_Safety_OK" stores this status for internal processing.
    Outcome:

    If "Safety_OK" = TRUE, the system is safe.
    If "Safety_OK" = FALSE, operations are blocked due to safety risks.

    2. Command Execution - Enabling Controls
    Inputs:

    "PB_VD" - Push button for door control.
    "CMD_Enable" - General command permission.
    Processing:

    "CMD_Enable" is set TRUE only if "PB_VD" is pressed AND the safety conditions ("Temp_Safety_OK") are met.
    "Temp_CMD_Enable" stores the temporary state of command permission.
    Outcome:

    If "CMD_Enable" = TRUE, machine commands are allowed.
    If "CMD_Enable" = FALSE, commands are blocked.

    3. Fault Detection & Memory (SR Latch)
    Faults Tracked:

    "Aux_SR_Fault" - Stores fault conditions until reset.
    "Temp_Fault" - Temporary fault variable.
    SR Latch Functionality:

    If "Temp_Fault" is TRUE, "Aux_SR_Fault" is latched (TRUE).
    "Aux_SR_Fault" stays TRUE until manually reset.
    Outcome:

    If a fault occurs, "Aux_SR_Fault" remains TRUE and requires manual reset before operations resume.
    The system prevents execution when "Aux_SR_Fault" is active.

    4. Temporary Control Variables - Internal Processing
    Variables Used:

    "Temp_CMD_Enable" - Stores intermediate command enable status.
    "Temp_Fault" - Stores temporary fault detection.
    "Temp_Safety_OK" - Used for internal safety checks.
    Processing:

    "Temp_Safety_OK" ensures "Safety_OK" is properly propagated.
    "Temp_Fault" prevents unsafe operations if a fault occurs.
    "Temp_CMD_Enable" manages command permissions internally.
    Outcome:

    If a fault occurs, "Temp_Fault" is triggered, blocking unsafe operations.
    If safety conditions are met, "Temp_Safety_OK" ensures safe system execution.

    Functionality:
    Safety checks ensure operations only run when the system is safe.
    Commands are only enabled if safety conditions are met.
    Faults are latched using "Aux_SR_Fault" and require manual reset.
    Temporary variables handle intermediate safety logic.


    xml files: {{snippets}}

    Query: {{query}}

    Memory: {{memory}}
    """

    prompt_fbd = """ 
    You are an expert to verify PLC programs in FBD.

    Your primary objective is to ensure the safety, reliability, and proper functionality of software used to an Automated People Mover, to be certified SIL 4 as per CENELEC standards.

    Your responses must be:
    - Clear, precise, and technically detailed.
    - Aligned with automated people mover standards and including EN 50128 as references.
            
    **Guidelines for Analysis**:
    - **Safety Priority**: Under no circumstances should you suggest modifications or enhancements that violate established safety principles, even if a requirement is found to be unmet.
    - **Thoroughness**: Analyze the program step-by-step to ensure a comprehensive understanding of its logic, structure, and functionality. Consider all the names of inputs, outputs, auxiliares, InOut, Temp, Return, Static, Network Names, Constants and comments to enhance your interpretation.
    - **Clarity**: If any part of the code or requirements is unclear or incomplete, specify what additional information is needed.

    The xml files are related to the control of {subject}. Receive the files and wait for the queries.

    For the complementary questions, after check of requirement, only reply directly about the additional question.

    Example of code interpretation:
    <SW.Blocks.CompileUnit xmlns="http://www.siemens.com/automation/Openness/SW/CompileUnit/v1">
    <SW.Blocks.CompileUnit.ID>1</SW.Blocks.CompileUnit.ID>
    <SW.Blocks.FBD>
        <Parts>

        <!-- Safety Logic -->
        <Network>
            <Comment>Ensure Safety Logic</Comment>
            <Part Name="OR" UId="10">
            <TemplateValue Name="Card" Type="Cardinality">2</TemplateValue>
            </Part>
            <Wire UId="20">
            <NameCon UId="10" Name="in1">
                <Component Name="Safety_Inp"/>
            </NameCon>
            <NameCon UId="10" Name="in2">
                <Component Name="Simulation"/>
            </NameCon>
            </Wire>
            <Wire UId="30">
            <NameCon UId="10" Name="out">
                <Component Name="Safety_OK"/>
            </NameCon>
            </Wire>
        </Network>

        <!-- Store Safety Status Temporarily -->
        <Network>
            <Comment>Store Safety Status Temporarily</Comment>
            <Wire UId="40">
            <NameCon UId="10" Name="out">
                <Component Name="Temp_Safety_OK"/>
            </NameCon>
            </Wire>
        </Network>

        <!-- Enable Command Execution -->
        <Network>
            <Comment>Enable Command Execution</Comment>
            <Part Name="AND" UId="50">
            <TemplateValue Name="Card" Type="Cardinality">2</TemplateValue>
            </Part>
            <Wire UId="60">
            <NameCon UId="50" Name="in1">
                <Component Name="PB_VD"/>
            </NameCon>
            <NameCon UId="50" Name="in2">
                <Component Name="Temp_Safety_OK"/>
            </NameCon>
            </Wire>
            <Wire UId="70">
            <NameCon UId="50" Name="out">
                <Component Name="CMD_Enable"/>
            </NameCon>
            </Wire>
        </Network>

        <!-- Temporary Command Enable -->
        <Network>
            <Comment>Temporary Command Enable</Comment>
            <Wire UId="80">
            <NameCon UId="70" Name="out">
                <Component Name="Temp_CMD_Enable"/>
            </NameCon>
            </Wire>
        </Network>

        <!-- Fault Detection Latch (SR) -->
        <Network>
            <Comment>Fault Detection Latch</Comment>
            <Part Name="SR" UId="90">
            <Symbol>
                <Component Name="Aux_SR_Fault"/>
            </Symbol>
            </Part>
            <Wire UId="100">
            <NameCon UId="90" Name="S">
                <Component Name="Temp_Fault"/>
            </NameCon>
            </Wire>
        </Network>

        <!-- Store Fault Condition -->
        <Network>
            <Comment>Store Fault Condition</Comment>
            <Wire UId="110">
            <NameCon UId="90" Name="Q">
                <Component Name="Temp_Fault"/>
            </NameCon>
            </Wire>
        </Network>

        <!-- Auxiliary Processing -->
        <Network>
            <Comment>Auxiliary Processing</Comment>
            <Wire UId="120">
            <NameCon UId="80" Name="out">
                <Component Name="Aux_SR_Close"/>
            </NameCon>
            </Wire>
        </Network>

        </Parts>
    </SW.Blocks.FBD>
    </SW.Blocks.CompileUnit>

    What Happens in the PLC?

    1. Safety Logic - Ensuring System Safety
    Inputs:

    "Safety_Inp" - Indicates if the safety system is active (TRUE = Safe).
    "Simulation" - Allows test mode activation (TRUE = Simulated mode).
    Processing:

    The OR Gate (UId=10) checks if either "Safety_Inp" or "Simulation" is TRUE.
    "Safety_OK" is set to TRUE if at least one of these conditions is met.
    Outcome:

    If "Safety_OK" = TRUE, the system operates normally.
    If "Safety_OK" = FALSE, operations are blocked due to a safety risk.

    2. Command Execution - Enabling Controls
    Inputs:

    "PB_VD" - A push button for door control.
    "CMD_Enable" - General command permission.
    Processing:

    The AND Gate (UId=50) ensures that "CMD_Enable" is activated only if "PB_VD" is pressed AND "Safety_OK" is TRUE.
    "CMD_Enable" is stored temporarily in "Temp_CMD_Enable".
    Outcome:

    If "CMD_Enable" = TRUE, machine commands are allowed.
    If "CMD_Enable" = FALSE, machine operations are blocked.

    3. Fault Detection & Memory (SR Latch)
    Faults Tracked:

    "Aux_SR_Fault" - Stores fault conditions until reset.
    "Temp_Fault" - Temporary fault variable.
    SR Latch Functionality:

    The SR Latch (UId=90) ensures that once a fault ("Temp_Fault") occurs, "Aux_SR_Fault" is latched (TRUE) and remains active until manually reset.
    Outcome:

    If "Temp_Fault" = TRUE, "Aux_SR_Fault" stays latched (TRUE).
    Operations cannot continue while a fault is active.

    4. Temporary Control Variables - Internal Processing
    Variables Used:

    "Temp_CMD_Enable" - Stores intermediate command enable status.
    "Temp_Fault" - Stores temporary fault detection.
    "Temp_Safety_OK" - Used for internal safety checks.
    Processing:

    "Temp_Safety_OK" ensures "Safety_OK" propagates correctly.
    "Temp_Fault" prevents operations if an issue exists.
    "Temp_CMD_Enable" manages command permissions internally.
    Outcome:

    If a fault is detected, "Temp_Fault" is triggered, preventing unsafe operations.
    If safety conditions are met, "Temp_Safety_OK" ensures safe system execution.

    Functionality:
    Safety conditions must be met before operations can proceed.
    Commands are only enabled if "PB_VD" is pressed and "Safety_OK" is TRUE.
    Faults are latched and must be manually reset.
    Temporary variables handle intermediate logic to control safety and execution.


    xml files: {{snippets}}

    Query: {{query}}

    Memory: {{memory}}
    """

    #Prompt selection condition
    if language == 'Ladder':
        prompt_str = prompt_ladder
    elif language == 'STL':
        prompt_str = prompt_stl
    elif language == 'CSL':
        prompt_str = prompt_scl
    else:
        prompt_str = prompt_fbd
    
    #Prompt definition
    prompt = ChatPromptTemplate.from_template(prompt_str)

    #Original chain
    chain = prompt | model | StrOutputParser()

    #Function to retrieve code snippets from query
    def retrieve_docs(query):
        snippets = retriever.invoke(query)        

        return snippets
    
    #Function for chat memory
    store = {}

    def get_session_id(session_id):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]
    
    #Memory chain
    memory_chain = RunnableWithMessageHistory(
    chain,
    get_session_id,
    input_messages_key = 'query',
    history_messages_key = 'memory',
    ) | StrOutputParser()

    #Configuration of session_id and user
    config = {'configurable': {'session_id': 'user_a'}}

    #Response management
    with st.spinner('AI Verifier working...'):
        def get_responses():

            if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []

            if subject:
                st.session_state.chat_history.append({'role': 'user', 'content': query})

                snippets = retrieve_docs(query)
                final_input = {'query': query, 'snippets': snippets}

                response = []

                #text_placeholder = st.empty()

                for chunk in memory_chain.stream(final_input, config = config):
                    response.append(chunk)

                    #text_placeholder.write(''.join(response))

                st.session_state.chat_history.append({'role': 'agent', 'content': ''.join(response)})
            
            st.write('Chat history:')

            for msg in st.session_state.chat_history:
                with st.container(border = True):
                    if msg['role'] == 'user':
                        text_part = st.info(f'User: {msg['content']}')
                    else:
                        text_part = st.markdown(f'**AI Verifier:** {msg['content']}')
        
            return text_part    
        
        if query := st.chat_input('Ask the Coester AI PLC Program Verifier:'):

            get_responses()