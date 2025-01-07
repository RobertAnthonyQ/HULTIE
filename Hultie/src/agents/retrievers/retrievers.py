from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv

#PUCP
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


# #----------------------------INDEX_DATABESES----------------------------------------------------------
# file_path="C:\\Users\diego\PycharmProjects\HultieChatbot\Hultie\data\PUCP_BASES.pdf"
# loader= PyPDFLoader(file_path)
# docs=loader.load()
# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=3000, chunk_overlap=30)
# doc_splits = text_splitter.split_documents(docs)
# embd = OpenAIEmbeddings()
# vectorstore = Chroma.from_documents(
#     documents=doc_splits,
#     collection_name="rag-pucp",
#     embedding=embd,
# )
# retriever_PUCP = vectorstore.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 3},
# )
#
# #UPC
# file_path="C:\\Users\diego\PycharmProjects\HultieChatbot\Hultie\data\\UPC_BASES.pdf"
# loader= PyPDFLoader(file_path)
# docs=loader.load()
# doc_splits = text_splitter.split_documents(docs)
# embd = OpenAIEmbeddings()
# vectorstore = Chroma.from_documents(
#     documents=doc_splits,
#     collection_name="rag-upc",
#     embedding=embd,
# )
# retriever_UPC = vectorstore.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 3},
# )
#
#
#
# #UP
# file_path="C:\\Users\diego\PycharmProjects\HultieChatbot\Hultie\data\\UP_BASES.pdf"
# loader= PyPDFLoader(file_path)
# docs=loader.load()
# doc_splits = text_splitter.split_documents(docs)
# embd = OpenAIEmbeddings()
# vectorstore = Chroma.from_documents(
#     documents=doc_splits,
#     collection_name="rag-up",
#     embedding=embd,
# )
# retriever_UP = vectorstore.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 3},
# )
#
# #ULIMA
# file_path="C:\\Users\diego\PycharmProjects\HultieChatbot\Hultie\data\\ULIMA_BASES.pdf"
# loader= PyPDFLoader(file_path)
# docs=loader.load()
# doc_splits = text_splitter.split_documents(docs)
# embd = OpenAIEmbeddings()
# vectorstore = Chroma.from_documents(
#     documents=doc_splits,
#     collection_name="rag-ulima",
#     embedding=embd,
# )
# retriever_ULIMA = vectorstore.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 3},
# )

retriever_UP = """
# 1. ¿Qué es hult prize? /Definición de hulprize, objetivo del concurso

Hult Prize es la competencia estudiantil más grande del mundo que inspira a jóvenes emprendedores a resolver los mayores desafíos globales mediante startups sostenibles. Se conoce como el “Premio Nobel de los Estudiantes”. Desde su fundación, más de 1 millón de estudiantes han participado, buscando generar impacto positivo alineado con los Objetivos de Desarrollo Sostenible (ODS) de la ONU.

Para esta edición (2025) el desafío es **“Unlimited”**: Los equipos pueden elegir resolver cualquier problema, siempre y cuando la startup esté alineada con al menos un ODS.

**Campus Director en la Universidad del Pacífico:** Brenda Jauregui

---

# 2. Elegibilidad y Registro

### Criterios de Elegibilidad:

**1. Estudiantes:**
- Deben estar matriculados en algún curso/ciclo/especialización universitaria (pregrado o posgrado) y tener al menos 18 años cumplidos al 28 de febrero de 2025.

**2. Equipos:**
- Cada equipo debe estar formado por **2 a 4 integrantes**.
- Los equipos pueden incluir estudiantes de diferentes facultades o incluso universidades, pero deben representar a una sola universidad, con al menos un miembro inscrito en ella.

> **Nota:** Todos los participantes deben estar registrados oficialmente en el formulario de equipo.

**3. Restricciones:**
- Los equipos se pueden inscribir en solo un programa OnCampus: representar a solo una universidad.
### Proceso de Registro:

1. Completar el formulario de registro oficial antes del **1 de febrero**.
2. Confirmación de registro: recibirán un correo electrónico con detalles adicionales para la participación.

---

# 3. Proceso del Concurso

### Fase Local - OnCampus (Enero - Febrero)

1. **Fase de Inscripción:**  
   Los equipos deberán registrarse antes del **1 de febrero de 2025**.

2. **Fase Preliminar:**
   - **Fecha:** 8 de febrero (fecha tentativa)  
   - **Formato:** En esta etapa, cada equipo contará con **1 minuto** para realizar su pitch, seguido de **1 minuto de preguntas** por parte del jurado. Los equipos serán evaluados mediante una clasificación por inversión simulada, en la cual se asignarán montos ficticios según el potencial del proyecto.  
     Los equipos cuyos montos de inversión sean los más altos pasarán a la siguiente fase.
   - **Actividades Previas de la fase preliminar:**  
     - Talleres y workshops enfocados en desarrollar habilidades relacionadas con la creación de ideas, modelos de negocio y estrategias de impacto social.  
     - Acceso a Hultie, un asistente virtual en WhatsApp que brinda retroalimentación y responde dudas frecuentes.

   > **Nota:** Asistencia obligatoria a los talleres por parte de al menos un miembro de cada equipo.

3. **Fase Semifinal:**
   - **Fecha:** 15 de febrero (fecha tentativa)  
   - **Formato:** En esta etapa, los equipos tendrán **3 minutos** para realizar su pitch, seguidos de **2 minutos de preguntas** del jurado. La evaluación continuará siendo mediante clasificación por inversión simulada, basada en el equipo y el potencial de los proyectos. Los equipos cuyos montos de inversión sean los más altos pasarán a la gran final.
   - **Actividades Previas de la fase semifinal:**  
     - Sesiones de mentoría con expertos para recibir retroalimentación y perfeccionar sus ideas.  
     - Talleres especializados en perfeccionamiento del modelo de negocio, impacto y viabilidad.
4. **Evento Final OnCampus**

- **Fecha y formato:** 22 de febrero (fecha tentativa) - Presencial.  
- Cada equipo tendrá **4 minutos** para presentar su pitch ante un panel de jueces expertos, seguido de **4 minutos de preguntas y respuestas**.  
- **Certificados:** Los equipos que completen su pitch y de los cuales al menos 1 integrante haya asistido a las mentorías y talleres recibirán el certificado oficial de participación.  
- **Selección:** El equipo ganador será anunciado al final del evento final de OnCampus y representará a la universidad en la **National Competition**.

---

### **National Competition (Mayo)**

Los ganadores de cada OnCampus competirán en la **National Competition**.  
Los equipos presentarán un pitch en inglés ante un panel de expertos. Serán **4 minutos de pitch** y **4 de preguntas**. El ganador nacional avanzará al **Hult Prize Digital Incubator**.

---

### **Digital Incubator (Junio - Julio)**

Hasta **60 startups globales** serán seleccionadas para participar en este programa intensivo, donde recibirán:

- Mentorías personalizadas con expertos globales.  
- Talleres sobre diseño de impacto, validación de mercado, desarrollo de MVP y estrategias de escalamiento.  
- Recursos educativos digitales y acceso a herramientas para desarrollar su empresa.

Al final del programa, las startups presentarán sus avances y los mejores equipos avanzarán al **Hult Prize Global Accelerator**.

### **Global Accelerator (Agosto)**

Hasta **25 startups** participarán en un programa presencial de un mes en Ashridge House, Reino Unido. Las startups recibirán:

- Acceso a inversionistas, socios potenciales y líderes de la industria.  
- Múltiples Demo Days para recibir retroalimentación directa y preparar sus propuestas finales.

Al final del programa, las startups presentarán sus avances y las mejores startups serán seleccionadas para competir en la **Global Final**.

---

### **Global Final (Septiembre)**

Los equipos finalistas presentarán sus proyectos ante un panel de jueces internacionales el **5 de septiembre** en Londres.  
El equipo ganador recibirá el gran premio de **$1 millón USD** para escalar su startup.

---

# 4. Criterios de Evaluación

Los jueces evaluarán las propuestas con base en cuatro categorías principales:

### **1. Equipo:**

- **Organización:** Roles definidos y claridad en responsabilidades.  
- **Colaboración:** Capacidad de trabajar juntos para desarrollar y presentar la idea.  
- **Experiencia y habilidades:** Competencias complementarias y alineadas con el proyecto.

- **Escala:**  
  0: Desorganizado, roles poco claros y/o falta de experiencia.  
  3: Equipo promedio con roles básicos y comunicación moderada.  
  5: Equipo cohesionado, roles claros, excelente comunicación y complementariedad.

---

### **2. Idea:**

- **Identificación del problema:** Comprensión clara del problema social o ambiental que se busca resolver.  
- **Solución innovadora:** Idea creativa y viable.  
- **Validación:** Existencia de pruebas, retroalimentación de usuarios o prototipos iniciales.

- **Escala:**  
  0: Problema poco definido, solución vaga y sin evidencias.  
  3: Problema claro, solución viable, con validación básica.
  5: Solución probada, con retroalimentación robusta y potencial de
impacto.
### **2. Impacto:**

- **Alineación:** Relación directa con al menos un ODS.  
- **Medición:** Definición clara de KPIs y métricas de impacto.  
- **Escalabilidad:** Potencial para expandir el impacto social a medida que crecen los ingresos.

- **Escala:**  
  0: Sin alineación con ODS ni métricas claras.  
  3: Alineación con ODS y métricas iniciales.  
  5: Impacto claro y escalable, con KPIs bien definidos.

---

### **3. Viabilidad del Negocio:**

- **Modelo de negocio:** Estructura clara, sostenible y realista.  
- **Economía unitaria:** Comprensión de los costos, ingresos y rentabilidad.  
- **Ventaja competitiva:** Elementos diferenciadores frente a otras soluciones.

- **Escala:**  
  0: Modelo débil y poco realista.  
  3: Modelo claro pero con áreas por fortalecer.  
  5: Modelo sólido, escalable y con alto potencial disruptivo.

---

### **5. Mentorías y Recursos**

- **Mentorías personalizadas:** Disponibles en febrero, con acceso a expertos locales e internacionales.  
- **Talleres exclusivos:** Sesiones en vivo sobre creación de modelos de negocio, habilidades para pitching, etc.  
- **Ponencias Nacionales:** Oportunidad de asistir a talleres de expertos reconocidos globalmente del ecosistema de innovación y startups.  
- **Material grabado:** Acceso a grabaciones de talleres y recursos adicionales durante todo el programa.

---

### **6. Premios**

1. **Primer Lugar en el OnCampus Program:**  
   - Avance directo a la Competencia Nacional Hult Prize.  
   - Certificado de excelencia.

2. **Otros Premios:**  
   - Certificados para todos los participantes que completen el programa.
### **7. Términos y Condiciones**

1. La participación implica la aceptación de estas bases y del **Código de Conducta Hult Prize**.  
2. Los participantes son responsables de los costos asociados con su participación (transporte, alimentación, etc.).  
3. Al menos un miembro del equipo ganador debe asistir presencialmente al evento final.  
4. Cualquier incumplimiento de las normas puede resultar en la descalificación inmediata.

---

### **8. Dudas y Consultas**

- **Correo:** by.jaureguir@alum.up.edu.pe  
- **Celular:** 947529791
"""

retriever_ULIMA="""
# 1. Introducción

Hult Prize es la competencia estudiantil más grande del mundo que inspira a jóvenes emprendedores a resolver los mayores desafíos globales mediante startups sostenibles. Se conoce como el “Premio Nobel de los Estudiantes”. Desde su fundación más de 1 millón de estudiantes han participado, buscando generar impacto positivo alineado con los Objetivos de Desarrollo Sostenible (ODS) de la ONU.

Para esta edición (2025) el desafío es **“Unlimited”**: Los equipos pueden elegir resolver cualquier problema, siempre y cuando la startup esté alineada con al menos un ODS.

**Campus Director en “Universidad de Lima”**: Belén Estefanía Salas Soto

---

# 2. Elegibilidad y Registro

## Criterios de Elegibilidad:

### 1. Estudiantes:
- **Deben estar matriculados en algún curso/ciclo/especialización universitaria (pregrado o posgrado) y tener al menos 18 años cumplidos al 28 de febrero de 2025.**

### 2. Equipos:
- **Cada equipo debe estar formado por 2 a 4 integrantes.**
- Los equipos pueden incluir estudiantes de diferentes facultades o incluso universidades, pero deben representar a una sola universidad, con al menos un miembro inscrito en ella.
- **Todos los participantes deben estar registrados oficialmente en el formulario del equipo.**

### 3. Restricciones:
- Los equipos se pueden inscribir en solo un programa OnCampus: representará a solo una universidad.

## Proceso de Registro:
1. Completar el formulario de registro oficial antes del 31 de enero.
2. Confirmación de registro: recibirán un correo electrónico con detalles adicionales para la participación.

---

# 3. Proceso del Concurso

## Fase Local - OnCampus (Enero - Febrero)

### 1. Fase de Inscripción:
Los equipos deben registrarse antes del 31 de enero de 2025.

### 2. Fase de Desarrollo:
- **Los equipos participarán en sesiones de mentoría y talleres durante febrero. La asistencia es obligatoria por al menos un miembro del equipo.**
- Tendrán la oportunidad de recibir retroalimentación directa para mejorar su modelo de negocio, impacto y pitch.
- Tendrán acceso a Hultie, el wsp agente que brinda retroalimentación y resuelve dudas.

### 3. Evento Final OnCampus:
- **Fecha y formato:** 22 de Marzo (fecha tentativa), presencial.
- Cada equipo tendrá 4 minutos para presentar su pitch ante un panel de jueces expertos, seguido de 4 minutos de preguntas y respuestas.
- **Certificados:** Los equipos que completen su pitch y de los que al menos 1 integrante haya asistido a las mentorías y talleres recibirán el certificado oficial de participación.
- **Selección:** El equipo ganador será anunciado al final del evento final del OnCampus y representará a la universidad en la National Competition.

---

## National Competition (Mayo)
Los ganadores de cada OnCampus competirán en la National Competition. Los equipos presentarán un pitch en inglés ante un panel de expertos. Serán 4 minutos de pitch y 4 de preguntas. El ganador nacional avanzará al **Hult Prize Digital Incubator**.

---

## Digital Incubator (Junio - Julio)
Hasta 60 startups globales serán seleccionadas para participar en este programa intensivo, donde recibirán:

- **Mentorías personalizadas con expertos globales.**
- **Talleres sobre diseño de impacto, validación de mercado, desarrollo de MVP y estrategias de escalamiento.**
- **Recursos educativos digitales y acceso a herramientas para desarrollar su empresa.**

Al final del programa, las startups presentarán sus avances y los mejores equipos avanzarán al **Hult Prize Global Accelerator**.

---

## Global Accelerator (Agosto)
Hasta 25 startups participarán en un programa presencial de un mes en Ashridge House, Reino Unido. Las startups recibirán:

- **Acceso a inversionistas, socios potenciales y líderes de la industria.**
- **Múltiples Demo Days para recibir retroalimentación directa y preparar sus propuestas finales.**

Al final del programa, las startups presentarán sus avances y las seis mejores startups serán seleccionadas para competir en la **Global Final**.

---

## Global Final (Septiembre)
Los seis equipos finalistas presentarán sus proyectos ante un panel de jueces internacionales el 5 de septiembre en Londres. El equipo ganador recibirá el gran premio de **$1 millón USD** para escalar su startup.

---

# 4. Criterios de Evaluación

Los jueces evaluarán las propuestas con base en cuatro categorías principales:

### 1. Equipo:
- **Organización:** Roles definidos y claridad en responsabilidades.
- **Colaboración:** Capacidad de trabajar juntos para desarrollar y presentar la idea.
- **Experiencia y habilidades:** Competencias complementarias y alineadas con el proyecto.

**Escala:**
- 0: Desorganizado, roles poco claros y/o falta de experiencia.
- 3: Equipo promedio con roles básicos y comunicación moderada.
- 5: Equipo cohesionado, roles claros, excelente comunicación y complementariedad.

---

### 2. Idea:
- **Identificación del problema:** Comprensión clara del problema social o ambiental que se busca resolver.
- **Solución innovadora:** Idea creativa y viable.
- **Validación:** Existencia de pruebas, retroalimentación de usuarios o prototipos iniciales.

**Escala:**
- 0: Problema poco definido, solución vaga y sin evidencia.
- 3: Problema claro, solución viable, con validación básica.
- 5: Solución probada, con retroalimentación robusta y potencial de impacto.

---

### 3. Impacto:
- **Alineación:** Relación directa con al menos un ODS.
- **Medición:** Definición clara de KPIs y métricas de impacto.
- **Escalabilidad:** Potencial para expandir el impacto social a medida que crecen los ingresos.

**Escala:**
- 0: Sin alineación con ODS ni métricas claras.
- 3: Alineación con ODS y métricas iniciales.
- 5: Impacto claro y escalable, con KPIs bien definidos.

---

### 4. Viabilidad del Negocio:
- **Modelo de negocio:** Estructura clara, sostenible y realista.
- **Economía unitaria:** Comprensión de los costos, ingresos y rentabilidad.
- **Ventaja competitiva:** Elementos diferenciadores frente a otras soluciones.

**Escala:**
- 0: Modelo débil y poco realista.
- 3: Modelo claro pero con áreas por fortalecer.
- 5: Modelo sólido, escalable y con alto potencial disruptivo.

---

# 5. Mentorías y Recursos

- **Mentorías personalizadas:** Disponibles en febrero, con acceso a expertos locales e internacionales.
- **Talleres exclusivos:** Sesiones en vivo sobre creación de modelos de negocio, habilidades para pitching, etc.
- **Ponencias Nacionales:** Oportunidad de asistir a talleres de expertos reconocidos globalmente del ecosistema de innovación y startups.
- **Material grabado:** Acceso a grabaciones de talleres y recursos adicionales durante todo el programa.

---

# 6. Términos y Condiciones

1. La participación implica la aceptación de estas bases y del **Código de Conducta Hult Prize**.
2. Los participantes son responsables de los costos asociados con su participación (transporte, alimentación, etc.).
3. Al menos un miembro del equipo ganador debe asistir presencialmente al evento final.
4. Cualquier incumplimiento de las normas puede resultar en la descalificación inmediata.

---

# Contacto

- **Correo electrónico:** belensalassoto2023@gmail.com  
- **WhatsApp:** 930684478
"""



retriever_CERTUS = """
# Bases del Concurso OnCampus Hult Prize  
### CERTUS

## 1. Introducción
Hult Prize es la competencia estudiantil más grande del mundo que inspira a jóvenes emprendedores a resolver los mayores desafíos globales mediante startups sostenibles. Se conoce como el “Premio Nobel de los Estudiantes”. Desde su fundación, más de 1 millón de estudiantes han participado, buscando generar impacto positivo alineado con los **Objetivos de Desarrollo Sostenible (ODS)** de la ONU.

Para esta edición (2025), el desafío es **“Unlimited”**: Los equipos pueden elegir resolver cualquier problema, siempre y cuando la startup esté alineada con al menos un ODS.

**Campus Director**: Equipo administrativo

---

## 2. Elegibilidad y Registro

### Criterios de Elegibilidad:

1. **Estudiantes**:
   - Deben estar matriculados en algún curso/ciclo/especialización universitaria (pregrado o posgrado) y tener al menos 18 años cumplidos al 28 de febrero de 2025.

2. **Equipos**:
   - Cada equipo debe estar formado por **2 a 4 integrantes**.
   - Los equipos pueden incluir estudiantes de diferentes facultades o incluso universidades, pero deben representar a una sola universidad, con al menos un miembro inscrito en ella.
   - Todos los participantes deben estar registrados oficialmente en el formulario del equipo.

3. **Restricciones**:
   - Los equipos se pueden inscribir en solo un programa OnCampus: representar a solo una universidad.

### Proceso de Registro:

1. Completar el formulario de registro oficial antes del **31 de enero**.  
2. Confirmación de registro: recibirán un correo electrónico con detalles adicionales para la participación.

---

## 3. Proceso del Concurso

### Fase Local - OnCampus (Enero - Febrero)
El equipo administrativo detallará el proceso solo a los inscritos. Para cualquier consulta, escribir a [hultprizeperuoficial@gmail.com](mailto:hultprizeperuoficial@gmail.com) con el asunto **“Consulta CERTUS”**.

### National Competition (Mayo)
Los ganadores de cada OnCampus competirán en la **National Competition**. Los equipos presentarán un pitch en inglés ante un panel de expertos. Serán **4 minutos** de pitch y **4 minutos** de preguntas. El ganador nacional avanzará al **Hult Prize Digital Incubator**.

### Digital Incubator (Junio - Julio)
Hasta **60 startups globales** serán seleccionadas para participar en este programa intensivo, donde recibirán:

- **Mentorías personalizadas** con expertos globales.
- **Talleres** sobre diseño de impacto, validación de mercado, desarrollo de MVP y estrategias de escalamiento.
- **Recursos educativos digitales** y acceso a herramientas para desarrollar su empresa.

Al final del programa, las startups presentarán sus avances y los mejores equipos avanzarán al **Hult Prize Global Accelerator**.

### Global Accelerator (Agosto)
Hasta **25 startups** participarán en un programa presencial de un mes en **Ashridge House, Reino Unido**. Las startups recibirán:

- Acceso a inversionistas, socios potenciales y líderes de la industria.
- Múltiples **Demo Days** para recibir retroalimentación directa y preparar sus propuestas finales.

Al final del programa, las startups presentarán sus avances y las seis mejores startups serán seleccionadas para competir en la **Global Final**.

### Global Final (Septiembre)
Los seis equipos finalistas presentarán sus proyectos ante un panel de jueces internacionales el **5 de septiembre en Londres**. El equipo ganador recibirá el gran premio de **$1 millón USD** para escalar su startup.

---

## 4. Criterios de Evaluación

Los jueces evaluarán las propuestas con base en cuatro categorías principales:

1. **Equipo**:
   - **Organización**: Roles definidos y claridad en responsabilidades.
   - **Colaboración**: Capacidad de trabajar juntos para desarrollar y presentar la idea.
   - **Experiencia y habilidades**: Competencias complementarias y alineadas con el proyecto.

2. **Idea**:
   - **Identificación del problema**: Comprensión clara del problema social o ambiental que se busca resolver.
   - **Solución innovadora**: Idea creativa y viable.
   - **Validación**: Existencia de pruebas, retroalimentación de usuarios o prototipos iniciales.

3. **Impacto**:
   - **Alineación**: Relación directa con al menos un ODS.
   - **Medición**: Definición clara de KPIs y métricas de impacto.
   - **Escalabilidad**: Potencial para expandir el impacto social a medida que crecen los ingresos.

4. **Viabilidad del Negocio**:
   - **Modelo de negocio**: Estructura clara, sostenible y realista.
   - **Economía unitaria**: Comprensión de los costos, ingresos y rentabilidad.
   - **Ventaja competitiva**: Elementos diferenciadores frente a otras soluciones.

---

## 5. Términos y Condiciones

1. La participación implica la aceptación de estas bases y del **Código de Conducta Hult Prize**.  
2. Los participantes son responsables de los costos asociados con su participación (transporte, alimentación, etc.).  
3. Al menos un miembro del equipo ganador debe asistir presencialmente al evento final.  
4. Cualquier incumplimiento de las normas puede resultar en la descalificación inmediata.

---

**Contacto**:  
[hultprizeperuoficial@gmail.com](mailto:hultprizeperuoficial@gmail.com)
"""
retriever_UCST = """
# Bases del Concurso OnCampus Hult Prize  
### Universidad Católica Santo Toribio | Chiclayo

## 1. Introducción
Hult Prize es la competencia estudiantil más grande del mundo que inspira a jóvenes emprendedores a resolver los mayores desafíos globales mediante startups sostenibles. Se conoce como el “Premio Nobel de los Estudiantes”. Desde su fundación, más de 1 millón de estudiantes han participado, buscando generar impacto positivo alineado con los **Objetivos de Desarrollo Sostenible (ODS)** de la ONU.

Para esta edición (2025), el desafío es **“Unlimited”**: Los equipos pueden elegir resolver cualquier problema, siempre y cuando la startup esté alineada con al menos un ODS.

**Campus Director**: Equipo administrativo

---

## 2. Elegibilidad y Registro

### Criterios de Elegibilidad:

1. **Estudiantes**:
   - Deben estar matriculados en algún curso/ciclo/especialización universitaria (pregrado o posgrado) y tener al menos 18 años cumplidos al 28 de febrero de 2025.

2. **Equipos**:
   - Cada equipo debe estar formado por **2 a 4 integrantes**.
   - Los equipos pueden incluir estudiantes de diferentes facultades o incluso universidades, pero deben representar a una sola universidad, con al menos un miembro inscrito en ella.
   - Todos los participantes deben estar registrados oficialmente en el formulario del equipo.

3. **Restricciones**:
   - Los equipos se pueden inscribir en solo un programa OnCampus: representar a solo una universidad.

### Proceso de Registro:

1. Completar el formulario de registro oficial antes del **31 de enero**.  
2. Confirmación de registro: recibirán un correo electrónico con detalles adicionales para la participación.

---

## 3. Proceso del Concurso

### Fase Local - OnCampus (Enero - Febrero)
El equipo administrativo detallará el proceso solo a los inscritos. Para cualquier consulta, escribir a [hultprizeperuoficial@gmail.com](mailto:hultprizeperuoficial@gmail.com) con el asunto **“Consulta Universidad Católica Santo Toribio”**.

### National Competition (Mayo)
Los ganadores de cada OnCampus competirán en la **National Competition**. Los equipos presentarán un pitch en inglés ante un panel de expertos. Serán **4 minutos** de pitch y **4 minutos** de preguntas. El ganador nacional avanzará al **Hult Prize Digital Incubator**.

### Digital Incubator (Junio - Julio)
Hasta **60 startups globales** serán seleccionadas para participar en este programa intensivo, donde recibirán:

- **Mentorías personalizadas** con expertos globales.
- **Talleres** sobre diseño de impacto, validación de mercado, desarrollo de MVP y estrategias de escalamiento.
- **Recursos educativos digitales** y acceso a herramientas para desarrollar su empresa.

Al final del programa, las startups presentarán sus avances y los mejores equipos avanzarán al **Hult Prize Global Accelerator**.

### Global Accelerator (Agosto)
Hasta **25 startups** participarán en un programa presencial de un mes en **Ashridge House, Reino Unido**. Las startups recibirán:

- Acceso a inversionistas, socios potenciales y líderes de la industria.
- Múltiples **Demo Days** para recibir retroalimentación directa y preparar sus propuestas finales.

Al final del programa, las startups presentarán sus avances y las seis mejores startups serán seleccionadas para competir en la **Global Final**.

### Global Final (Septiembre)
Los seis equipos finalistas presentarán sus proyectos ante un panel de jueces internacionales el **5 de septiembre en Londres**. El equipo ganador recibirá el gran premio de **$1 millón USD** para escalar su startup.

---

## 4. Criterios de Evaluación

Los jueces evaluarán las propuestas con base en cuatro categorías principales:

1. **Equipo**:
   - **Organización**: Roles definidos y claridad en responsabilidades.
   - **Colaboración**: Capacidad de trabajar juntos para desarrollar y presentar la idea.
   - **Experiencia y habilidades**: Competencias complementarias y alineadas con el proyecto.

2. **Idea**:
   - **Identificación del problema**: Comprensión clara del problema social o ambiental que se busca resolver.
   - **Solución innovadora**: Idea creativa y viable.
   - **Validación**: Existencia de pruebas, retroalimentación de usuarios o prototipos iniciales.

3. **Impacto**:
   - **Alineación**: Relación directa con al menos un ODS.
   - **Medición**: Definición clara de KPIs y métricas de impacto.
   - **Escalabilidad**: Potencial para expandir el impacto social a medida que crecen los ingresos.

4. **Viabilidad del Negocio**:
   - **Modelo de negocio**: Estructura clara, sostenible y realista.
   - **Economía unitaria**: Comprensión de los costos, ingresos y rentabilidad.
   - **Ventaja competitiva**: Elementos diferenciadores frente a otras soluciones.

---

## 5. Términos y Condiciones

1. La participación implica la aceptación de estas bases y del **Código de Conducta Hult Prize**.  
2. Los participantes son responsables de los costos asociados con su participación (transporte, alimentación, etc.).  
3. Al menos un miembro del equipo ganador debe asistir presencialmente al evento final.  
4. Cualquier incumplimiento de las normas puede resultar en la descalificación inmediata.

---

**Contacto**:  
[hultprizeperuoficial@gmail.com](mailto:hultprizeperuoficial@gmail.com)

"""
retriever_UPC = """
# Bases del Concurso OnCampus Hult Prize  
### Universidad Peruana de Ciencias Aplicadas (UPC)

## 1. Introducción
Hult Prize es la competencia estudiantil más grande del mundo que inspira a jóvenes emprendedores a resolver los mayores desafíos globales mediante startups sostenibles. Se conoce como el “Premio Nobel de los Estudiantes”. Desde su fundación, más de 1 millón de estudiantes han participado, buscando generar impacto positivo alineado con los **Objetivos de Desarrollo Sostenible (ODS)** de la ONU.

Para esta edición (2025), el desafío es **“Unlimited”**: Los equipos pueden elegir resolver cualquier problema, siempre y cuando la startup esté alineada con al menos un ODS.

**Campus Director**: Equipo administrativo

---

## 2. Elegibilidad y Registro

### Criterios de Elegibilidad:

1. **Estudiantes**:
   - Deben estar matriculados en algún curso/ciclo/especialización universitaria (pregrado o posgrado) y tener al menos 18 años cumplidos al 28 de febrero de 2025.

2. **Equipos**:
   - Cada equipo debe estar formado por **2 a 4 integrantes**.
   - Los equipos pueden incluir estudiantes de diferentes facultades o incluso universidades, pero deben representar a una sola universidad, con al menos un miembro inscrito en ella.
   - Todos los participantes deben estar registrados oficialmente en el formulario del equipo.

3. **Restricciones**:
   - Los equipos se pueden inscribir en solo un programa OnCampus: representar a solo una universidad.

### Proceso de Registro:

1. Completar el formulario de registro oficial antes del **31 de enero**.  
2. Confirmación de registro: recibirán un correo electrónico con detalles adicionales para la participación.

---

## 3. Proceso del Concurso

### Fase Local - OnCampus (Enero - Febrero)
El equipo administrativo detallará el proceso solo a los inscritos. Para cualquier consulta, escribir a [hultprizeperuoficial@gmail.com](mailto:hultprizeperuoficial@gmail.com) con el asunto **“Universidad Peruana de Ciencias Aplicadas”**.

### National Competition (Mayo)
Los ganadores de cada OnCampus competirán en la **National Competition**. Los equipos presentarán un pitch en inglés ante un panel de expertos. Serán **4 minutos** de pitch y **4 minutos** de preguntas. El ganador nacional avanzará al **Hult Prize Digital Incubator**.

### Digital Incubator (Junio - Julio)
Hasta **60 startups globales** serán seleccionadas para participar en este programa intensivo, donde recibirán:

- **Mentorías personalizadas** con expertos globales.
- **Talleres** sobre diseño de impacto, validación de mercado, desarrollo de MVP y estrategias de escalamiento.
- **Recursos educativos digitales** y acceso a herramientas para desarrollar su empresa.

Al final del programa, las startups presentarán sus avances y los mejores equipos avanzarán al **Hult Prize Global Accelerator**.

### Global Accelerator (Agosto)
Hasta **25 startups** participarán en un programa presencial de un mes en **Ashridge House, Reino Unido**. Las startups recibirán:

- Acceso a inversionistas, socios potenciales y líderes de la industria.
- Múltiples **Demo Days** para recibir retroalimentación directa y preparar sus propuestas finales.

Al final del programa, las startups presentarán sus avances y las seis mejores startups serán seleccionadas para competir en la **Global Final**.

### Global Final (Septiembre)
Los seis equipos finalistas presentarán sus proyectos ante un panel de jueces internacionales el **5 de septiembre en Londres**. El equipo ganador recibirá el gran premio de **$1 millón USD** para escalar su startup.

---

## 4. Criterios de Evaluación

Los jueces evaluarán las propuestas con base en cuatro categorías principales:

1. **Equipo**:
   - **Organización**: Roles definidos y claridad en responsabilidades.
   - **Colaboración**: Capacidad de trabajar juntos para desarrollar y presentar la idea.
   - **Experiencia y habilidades**: Competencias complementarias y alineadas con el proyecto.

2. **Idea**:
   - **Identificación del problema**: Comprensión clara del problema social o ambiental que se busca resolver.
   - **Solución innovadora**: Idea creativa y viable.
   - **Validación**: Existencia de pruebas, retroalimentación de usuarios o prototipos iniciales.

3. **Impacto**:
   - **Alineación**: Relación directa con al menos un ODS.
   - **Medición**: Definición clara de KPIs y métricas de impacto.
   - **Escalabilidad**: Potencial para expandir el impacto social a medida que crecen los ingresos.

4. **Viabilidad del Negocio**:
   - **Modelo de negocio**: Estructura clara, sostenible y realista.
   - **Economía unitaria**: Comprensión de los costos, ingresos y rentabilidad.
   - **Ventaja competitiva**: Elementos diferenciadores frente a otras soluciones.

---

## 5. Términos y Condiciones

1. La participación implica la aceptación de estas bases y del **Código de Conducta Hult Prize**.  
2. Los participantes son responsables de los costos asociados con su participación (transporte, alimentación, etc.).  
3. Al menos un miembro del equipo ganador debe asistir presencialmente al evento final.  
4. Cualquier incumplimiento de las normas puede resultar en la descalificación inmediata.

---

**Contacto**:  
[hultprizeperuoficial@gmail.com](mailto:hultprizeperuoficial@gmail.com)

"""
retriever_UPCH = """
# Bases del Concurso OnCampus Hult Prize  
## UPCH  

---

## 1. Introducción

Hult Prize es la competencia estudiantil más grande del mundo que inspira a jóvenes emprendedores a resolver los mayores desafíos globales mediante startups sostenibles. Se conoce como el “Premio Nobel de los Estudiantes”. Desde su fundación más de 1 millón de estudiantes han participado, buscando generar impacto positivo alineado con los Objetivos de Desarrollo Sostenible (ODS) de la ONU.

Para esta edición (2025) el desafío es **“Unlimited”**: Los equipos pueden elegir resolver cualquier problema, siempre y cuando la startup esté alineada con al menos un ODS.

**Campus Director en UPCH:** Joseph Jesus Melgarejo Castillo

---

## 2. Elegibilidad y Registro

### **Criterios de Elegibilidad**

1. **Estudiantes:**  
   - Deben estar matriculados en algún curso/ciclo/especialización universitaria (pregrado o posgrado) y tener al menos 18 años cumplidos al 28 de febrero de 2025.

2. **Equipos:**  
   - Cada equipo debe estar formado por **2 a 4 integrantes**.  
   - Los equipos pueden incluir estudiantes de diferentes facultades o incluso universidades, pero deben representar a una sola universidad, con al menos un miembro inscrito en ella.  
   - Todos los participantes deben estar registrados oficialmente en el formulario del equipo.

3. **Restricciones:**  
   - Los equipos se pueden inscribir en solo un programa OnCampus: representar a solo una universidad.

### **Proceso de Registro**

1. Completar el formulario de registro oficial antes del **25 de febrero**.  
2. Confirmación de registro: recibirán un correo electrónico con detalles adicionales para la participación.

---

## 3. Proceso del Concurso

### **Fase Local - OnCampus (Enero - Febrero)**

#### 1. Fase de Inscripción  
Los equipos deberán registrarse antes del **31 de enero de 2025**.

#### 2. Fase de Desarrollo  
- Los equipos participarán en sesiones de mentoría y talleres durante febrero. La asistencia es obligatoria por al menos un miembro del equipo.  
- Tendrán la oportunidad de recibir retroalimentación directa para mejorar su modelo de negocio, impacto y pitch.  
- Tendrán acceso a **Hultie**, el wsp agente que brinda retroalimentación y resuelve dudas frecuentes.

#### 3. Evento Final OnCampus  
- **Fecha y formato:** Viernes **28 de Febrero del 2025** (tentativo), presencial.  
- Cada equipo tendrá 4 minutos para presentar su pitch ante un panel de jueces expertos, seguido de 4 minutos de preguntas y respuestas.  
- **Certificados:** Los equipos que completen su pitch y de los que al menos 1 integrante haya asistido a las mentorías y talleres recibirán el certificado oficial de participación.  
- **Selección:** El equipo ganador será anunciado al final del evento final del OnCampus y representará a la universidad en la **National Competition**.

---

### **National Competition (Mayo)**

Los ganadores de cada OnCampus competirán en la **National Competition**. Los equipos presentarán un pitch en inglés ante un panel de expertos. Serán 4 minutos de pitch y 4 de preguntas. El ganador nacional avanzará al **Hult Prize Digital Incubator**.

---

### **Digital Incubator (Junio - Julio)**

Hasta 60 startups globales serán seleccionadas para participar en este programa intensivo, donde recibirán:

- Mentorías personalizadas con expertos globales.  
- Talleres sobre diseño de impacto, validación de mercado, desarrollo de MVP y estrategias de escalamiento.  
- Recursos educativos digitales y acceso a herramientas para desarrollar su empresa.

Al final del programa, las startups presentarán sus avances y los mejores equipos avanzarán al **Hult Prize Global Accelerator**.

---

### **Global Accelerator (Agosto)**

Hasta 25 startups participarán en un programa presencial de un mes en Ashridge House, Reino Unido. Las startups recibirán:

- Acceso a inversionistas, socios potenciales y líderes de la industria.  
- Múltiples Demo Days para recibir retroalimentación directa y preparar sus propuestas finales.

Al final del programa, las startups presentarán sus avances y las seis mejores startups serán seleccionadas para competir en la **Global Final**.

---

### **Global Final (Septiembre)**

Los seis equipos finalistas presentarán sus proyectos ante un panel de jueces internacionales el **5 de septiembre en Londres**. El equipo ganador recibirá el gran premio de **$1 millón USD** para escalar su startup.

---

## 4. Criterios de Evaluación

Los jueces evaluarán las propuestas con base en cuatro categorías principales:

### **1. Equipo**

- **Organización:** Roles definidos y claridad en responsabilidades.  
- **Colaboración:** Capacidad de trabajar juntos para desarrollar y presentar la idea.  
- **Experiencia y habilidades:** Competencias complementarias y alineadas con el proyecto.  

**Escala:**  
- 1: Desorganizado, roles poco claros y/o falta de experiencia.  
- 3: Equipo promedio con roles básicos y comunicación moderada.  
- 5: Equipo cohesionado, roles claros, excelente comunicación y complementariedad.

---

### **2. Idea**

- **Identificación del problema:** Comprensión clara del problema social o ambiental que se busca resolver.  
- **Solución innovadora:** Idea creativa y viable.  
- **Validación:** Existencia de pruebas, retroalimentación de usuarios o prototipos iniciales.  

**Escala:**  
- 1: Problema poco definido, solución vaga y sin evidencia.  
- 3: Problema claro, solución viable, con validación básica.  
- 5: Solución probada, con retroalimentación robusta y potencial de impacto.

---

### **3. Impacto**

- **Alineación:** Relación directa con al menos un ODS.  
- **Medición:** Definición clara de KPIs y métricas de impacto.  
- **Escalabilidad:** Potencial para expandir el impacto social a medida que crecen los ingresos.

**Escala:**  
- 1: Sin alineación con ODS ni métricas claras.  
- 3: Alineación con ODS y métricas iniciales.  
- 5: Impacto claro y escalable, con KPIs bien definidos.

---

### **4. Viabilidad del Negocio**

- **Modelo de negocio:** Estructura clara, sostenible y realista.  
- **Economía unitaria:** Comprensión de los costos, ingresos y rentabilidad.  
- **Ventaja competitiva:** Elementos diferenciadores frente a otras soluciones.

**Escala:**  
- 1: Modelo débil y poco realista.  
- 3: Modelo claro pero con áreas por fortalecer.  
- 5: Modelo sólido, escalable y con alto potencial disruptivo.

---

## 5. Mentorías y Recursos

- **Mentorías personalizadas:** Disponibles en febrero, con acceso a expertos locales e internacionales.  
- **Talleres exclusivos:** Sesiones en vivo sobre creación de modelos de negocio, habilidades para pitching, etc.  
- **Ponencias Nacionales:** Oportunidad de asistir a talleres de expertos reconocidos globalmente del ecosistema de innovación y startups.  
- **Material grabado:** Acceso a grabaciones de talleres y recursos adicionales durante todo el programa.

---

## 6. Premios

1. **Primer Lugar en el OnCampus Program:**  
   - Avance directo a la Competencia Nacional Hult Prize.  
   - Certificado de excelencia.

2. **Otros premios:**  
   - Certificados para todos los participantes que completen el programa.

---

## 7. Términos y Condiciones

1. La participación implica la aceptación de estas bases y del **Código de Conducta Hult Prize**.  
2. Los participantes son responsables de los costos asociados con su participación (transporte, alimentación, etc.).  
3. Al menos un miembro del equipo ganador debe asistir presencialmente al evento final.  
4. Cualquier incumplimiento de las normas puede resultar en la descalificación inmediata.

---

## **Contacto**

- **Correo:** [upchhultprize@gmail.com]  
- **Teléfono:** [982491910]
"""
retriever_UNCP = """
# Bases del Concurso OnCampus Hult Prize
**Universidad Nacional del Centro del Perú, Huancayo**

## 1. Introducción
Hult Prize es la competencia estudiantil más grande del mundo que inspira a jóvenes emprendedores a resolver los mayores desafíos globales mediante startups sostenibles. Se conoce como el “Premio Nobel de los Estudiantes”. Desde su fundación más de 1 millón de estudiantes han participado, buscando generar impacto positivo alineado con los Objetivos de Desarrollo Sostenible (ODS) de la ONU.

Para esta edición (2025) el desafío es **“Unlimited”**: Los equipos pueden elegir resolver cualquier problema, siempre y cuando la startup esté alineada con al menos un ODS.

**Campus Director**: Mabell Canchanya

---

## 2. Elegibilidad y Registro

### Criterios de Elegibilidad:
1. **Estudiantes**:
   - Deben estar matriculados en algún curso/ciclo/especialización universitaria (pregrado o posgrado) y tener al menos 18 años cumplidos al 28 de febrero de 2025.
   
2. **Equipos**:
   - Cada equipo debe estar formado por **2 a 4 integrantes**.
   - Los equipos pueden incluir estudiantes de diferentes facultades o incluso universidades, pero deben representar a una sola universidad, con al menos un miembro inscrito en ella.
   - Todos los participantes deben estar registrados oficialmente en el formulario del equipo.
   
3. **Restricciones**:
   - Los equipos se pueden inscribir en solo un programa OnCampus: representar a solo una universidad.

### Proceso de Registro:
1. Completar el **formulario de registro oficial** antes del **31 de enero**.
2. Confirmación de registro: recibirán un correo electrónico con detalles adicionales para la participación.

---

## 3. Proceso del Concurso

### Fase Local - OnCampus (Enero - Febrero)

1. **Fase de Inscripción**:
   - Los equipos deberán registrarse antes del **31 de enero de 2025**.

2. **Fase de Desarrollo**:
   - Los equipos participarán en sesiones de mentoría y talleres durante febrero. La asistencia es obligatoria por al menos un miembro del equipo.
   - Tendrán la oportunidad de recibir retroalimentación directa para mejorar su modelo de negocio, impacto y pitch.
   - Tendrán acceso a **Hultie**, el wsp agent que brinda retroalimentación y resuelve dudas frecuentes.

3. **Evento Final OnCampus**:
   - **Fecha y formato**: [detallar], presencial.
   - Cada equipo tendrá **4 minutos para presentar su pitch** ante un panel de jueces expertos, seguido de **4 minutos de preguntas y respuestas**.
   - **Certificados**: Los equipos que completen su pitch y de los que al menos 1 integrante haya asistido a las mentorías y talleres recibirán el certificado oficial de participación.
   - **Selección**: El equipo ganador será anunciado al final del evento final del OnCampus y representará a la universidad en la **National Competition**.

### National Competition (Mayo)
Los ganadores de cada OnCampus competirán en la **National Competition**. Los equipos presentarán un pitch en inglés ante un panel de expertos. Serán **4 minutos de pitch y 4 de preguntas**. El ganador nacional avanzará al **Hult Prize Digital Incubator**.

### Digital Incubator (Junio - Julio)
Hasta **60 startups globales** serán seleccionadas para participar en este programa intensivo, donde recibirán:
- **Mentorías personalizadas con expertos globales**.
- **Talleres sobre diseño de impacto, validación de mercado, desarrollo de MVP y estrategias de escalamiento**.
- **Recursos educativos digitales y acceso a herramientas para desarrollar su empresa**.

Al final del programa, las startups presentarán sus avances y los mejores equipos avanzarán al **Hult Prize Global Accelerator**.

### Global Accelerator (Agosto)
Hasta **25 startups** participarán en un programa presencial de un mes en **Ashridge House, Reino Unido**. Las startups recibirán:
- **Acceso a inversionistas, socios potenciales y líderes de la industria**.
- **Múltiples Demo Days** para recibir retroalimentación directa y preparar sus propuestas finales.

Al final del programa, las startups presentarán sus avances y las seis mejores startups serán seleccionadas para competir en la **Global Final**.

### Global Final (Septiembre)
Los seis equipos finalistas presentarán sus proyectos ante un panel de jueces internacionales el **5 de septiembre en Londres**. El equipo ganador recibirá el gran premio de **$1 millón USD** para escalar su startup.

---

## 4. Criterios de Evaluación

Los jueces evaluarán las propuestas con base en cuatro categorías principales:

1. **Equipo**:
   - **Organización**: Roles definidos y claridad en responsabilidades.
   - **Colaboración**: Capacidad de trabajar juntos para desarrollar y presentar la idea.
   - **Experiencia y habilidades**: Competencias complementarias y alineadas con el proyecto.
   - **Escala**:
     - **1**: Desorganizado, roles poco claros y/o falta de experiencia.
     - **3**: Equipo promedio con roles básicos y comunicación moderada.
     - **5**: Equipo cohesionado, roles claros, excelente comunicación y complementariedad.

2. **Idea**:
   - **Identificación del problema**: Comprensión clara del problema social o ambiental que se busca resolver.
   - **Solución innovadora**: Idea creativa y viable.
   - **Validación**: Existencia de pruebas, retroalimentación de usuarios o prototipos iniciales.
   - **Escala**:
     - **1**: Problema poco definido, solución vaga y sin evidencia.
     - **3**: Problema claro, solución viable, con validación básica.
     - **5**: Solución probada, con retroalimentación robusta y potencial de impacto.

3. **Impacto**:
   - **Alineación**: Relación directa con al menos un ODS.
   - **Medición**: Definición clara de KPIs y métricas de impacto.
   - **Escalabilidad**: Potencial para expandir el impacto social a medida que crecen los ingresos.
   - **Escala**:
     - **1**: Sin alineación con ODS ni métricas claras.
     - **3**: Alineación con ODS y métricas iniciales.
     - **5**: Impacto claro y escalable, con KPIs bien definidos.

4. **Viabilidad del Negocio**:
   - **Modelo de negocio**: Estructura clara, sostenible y realista.
   - **Economía unitaria**: Comprensión de los costos, ingresos y rentabilidad.
   - **Ventaja competitiva**: Elementos diferenciadores frente a otras soluciones.
   - **Escala**:
     - **1**: Modelo débil y poco realista.
     - **3**: Modelo claro pero con áreas por fortalecer.
     - **5**: Modelo sólido, escalable y con alto potencial disruptivo.

---

## 5. Mentorías y Recursos
- **Mentorías personalizadas**: Disponibles en febrero, con acceso a expertos locales e internacionales.
- **Talleres exclusivos**: Sesiones en vivo sobre creación de modelos de negocio, habilidades para pitching, etc.
- **Ponencias Nacionales**: Oportunidad de asistir a talleres de expertos reconocidos globalmente del ecosistema de innovación y startups.
- **Material grabado**: Acceso a grabaciones de talleres y recursos adicionales durante todo el programa.

---

## 7. Términos y Condiciones
1. La participación implica la aceptación de estas bases y del **Código de Conducta Hult Prize**.
2. Los participantes son responsables de los costos asociados con su participación (transporte, alimentación, etc.).
3. Al menos un miembro del equipo ganador debe asistir presencialmente al evento final.
4. Cualquier incumplimiento de las normas puede resultar en la descalificación inmediata.

"""
retriever_UNMSM = """
# Bases del Concurso OnCampus Hult Prize  
### UNMSM  

---

## Introducción

Hult Prize es la competencia estudiantil más grande del mundo que inspira a jóvenes emprendedores a resolver los mayores desafíos globales mediante startups sostenibles. Se conoce como el “Premio Nobel de los Estudiantes”. Desde su fundación más de 1 millón de estudiantes han participado, buscando generar impacto positivo alineado con los Objetivos de Desarrollo Sostenible (ODS) de la ONU.

Para esta edición (2025) el desafío es **“Unlimited”**: Los equipos pueden elegir resolver cualquier problema, siempre y cuando la startup esté alineada con al menos un ODS.

**Campus Director en UNMSM:** Alessandra Igmizu

---

## Elegibilidad y Registro

### **Criterios de Elegibilidad**

1. **Estudiantes:**  
   - Deben estar matriculados en algún curso/ciclo/especialización universitaria (pregrado o posgrado) y tener al menos 18 años cumplidos al 28 de febrero de 2025.

2. **Equipos:**  
   - Cada equipo debe estar formado por **2 a 4 integrantes**.  
   - Los equipos pueden incluir estudiantes de diferentes facultades o incluso universidades, pero deben representar a una sola universidad, con al menos un miembro inscrito en ella.  
   - Todos los participantes deben estar registrados oficialmente en el formulario del equipo.

3. **Restricciones:**  
   - Los equipos se pueden inscribir en solo un programa OnCampus: representar a solo una universidad.

---

### **Proceso de Registro**

1. Completar el formulario de registro oficial antes del **31 de enero**.  
2. Confirmación de registro: recibirán un correo electrónico con detalles adicionales para la participación.

---

## Proceso del Concurso

### **Fase Local - OnCampus (Enero - Febrero)**

#### **1. Fase de Inscripción**  
Los equipos deberán registrarse antes del **31 de enero de 2025**.

#### **2. Fase de Desarrollo**  
- Los equipos participarán en sesiones de mentoría y talleres durante febrero. La asistencia es obligatoria por al menos un miembro del equipo.  
- Tendrán la oportunidad de recibir retroalimentación directa para mejorar su modelo de negocio, impacto y pitch.  
- Tendrán acceso a **Hultie**, el WhatsApp agent que brinda retroalimentación y resuelve dudas frecuentes.

#### **3. Evento Final OnCampus**  
- **Fecha y formato:** Viernes **21 de febrero** (fecha tentativa), presencial.  
- Cada equipo tendrá 4 minutos para presentar su pitch ante un panel de jueces expertos, seguido de 4 minutos de preguntas y respuestas.  
- **Certificados:** Los equipos que completen su pitch y de los que al menos 1 integrante haya asistido a las mentorías y talleres recibirán el certificado oficial de participación.  
- **Selección:** El equipo ganador será anunciado al final del evento final del OnCampus y representará a la universidad en la **National Competition**.

---

### **National Competition (Mayo)**

Los ganadores de cada OnCampus competirán en la **National Competition**. Los equipos presentarán un pitch en inglés ante un panel de expertos. Serán 4 minutos de pitch y 4 de preguntas. El ganador nacional avanzará al **Hult Prize Digital Incubator**.

---

### **Digital Incubator (Junio - Julio)**

Hasta 60 startups globales serán seleccionadas para participar en este programa intensivo, donde recibirán:

- Mentorías personalizadas con expertos globales.  
- Talleres sobre diseño de impacto, validación de mercado, desarrollo de MVP y estrategias de escalamiento.  
- Recursos educativos digitales y acceso a herramientas para desarrollar su empresa.

Al final del programa, las startups presentarán sus avances y los mejores equipos avanzarán al **Hult Prize Global Accelerator**.

---

### **Global Accelerator (Agosto)**

Hasta 25 startups participarán en un programa presencial de un mes en Ashridge House, Reino Unido. Las startups recibirán:

- Acceso a inversionistas, socios potenciales y líderes de la industria.  
- Múltiples Demo Days para recibir retroalimentación directa y preparar sus propuestas finales.

Al final del programa, las startups presentarán sus avances y las seis mejores startups serán seleccionadas para competir en la **Global Final**.

---

### **Global Final (Septiembre)**

Los seis equipos finalistas presentarán sus proyectos ante un panel de jueces internacionales el **5 de septiembre en Londres**. El equipo ganador recibirá el gran premio de **$1 millón USD** para escalar su startup.

---

## Criterios de Evaluación

Los jueces evaluarán las propuestas con base en cuatro categorías principales:

### **1. Equipo**

- **Organización:** Roles definidos y claridad en responsabilidades.  
- **Colaboración:** Capacidad de trabajar juntos para desarrollar y presentar la idea.  
- **Experiencia y habilidades:** Competencias complementarias y alineadas con el proyecto.

**Escala:**  
- 1: Desorganizado, roles poco claros y/o falta de experiencia.  
- 3: Equipo promedio con roles básicos y comunicación moderada.  
- 5: Equipo cohesionado, roles claros, excelente comunicación y complementariedad.

---

### **2. Idea**

- **Identificación del problema:** Comprensión clara del problema social o ambiental que se busca resolver.  
- **Solución innovadora:** Idea creativa y viable.  
- **Validación:** Existencia de pruebas, retroalimentación de usuarios o prototipos iniciales.

**Escala:**  
- 1: Problema poco definido, solución vaga y sin evidencia.  
- 3: Problema claro, solución viable, con validación básica.  
- 5: Solución probada, con retroalimentación robusta y potencial de impacto.

---

### **3. Impacto**

- **Alineación:** Relación directa con al menos un ODS.  
- **Medición:** Definición clara de KPIs y métricas de impacto.  
- **Escalabilidad:** Potencial para expandir el impacto social a medida que crecen los ingresos.

**Escala:**  
- 1: Sin alineación con ODS ni métricas claras.  
- 3: Alineación con ODS y métricas iniciales.  
- 5: Impacto claro y escalable, con KPIs bien definidos.

---

### **4. Viabilidad del Negocio**

- **Modelo de negocio:** Estructura clara, sostenible y realista.  
- **Economía unitaria:** Comprensión de los costos, ingresos y rentabilidad.  
- **Ventaja competitiva:** Elementos diferenciadores frente a otras soluciones.

**Escala:**  
- 1: Modelo débil y poco realista.  
- 3: Modelo claro pero con áreas por fortalecer.  
- 5: Modelo sólido, escalable y con alto potencial disruptivo.

---

## Mentorías y Recursos

- **Mentorías personalizadas:** Disponibles en febrero, con acceso a expertos locales e internacionales.  
- **Talleres exclusivos:** Sesiones en vivo sobre creación de modelos de negocio, habilidades para pitching, etc.  
- **Ponencias Nacionales:** Oportunidad de asistir a talleres de expertos reconocidos globalmente del ecosistema de innovación y startups.  
- **Material grabado:** Acceso a grabaciones de talleres y recursos adicionales durante todo el programa.

---

## Premios

1. **Primer Lugar en el OnCampus Program:**  
   - Avance directo a la Competencia Nacional Hult Prize.  
   - Certificado de excelencia.

2. **Otros premios:**  
   - Certificados para todos los participantes que completen el programa.

---

## Términos y Condiciones

1. La participación implica la aceptación de estas bases y del **Código de Conducta Hult Prize**.  
2. Los participantes son responsables de los costos asociados con su participación (transporte, alimentación, etc.).  
3. Al menos un miembro del equipo ganador debe asistir presencialmente al evento final.  
4. Cualquier incumplimiento de las normas puede resultar en la descalificación inmediata.

---

## Contacto

- **Correo:** ale.igmizu@gmail.com  
- **Celular:** 957130856  
- **LinkedIn:** [Alessandra Igmizu](https://www.linkedin.com/in/alessandra-igmizu/)  
- **Hult Prize UNMSM:** hultprizeatunmsm@gmail.com
"""
retriever_UNAP = """
# Bases del Concurso OnCampus Hult Prize  
**Universidad Nacional de la Amazonía Peruana (UNAP)**

---

## 1. Introducción

Hult Prize es la competencia estudiantil más grande del mundo que inspira a jóvenes emprendedores a resolver los mayores desafíos globales mediante startups sostenibles. Se conoce como el “Premio Nobel de los Estudiantes”. Desde su fundación más de 1 millón de estudiantes han participado, buscando generar impacto positivo alineado con los Objetivos de Desarrollo Sostenible (ODS) de la ONU.

Para esta edición (2025) el desafío es **“Unlimited”**: Los equipos pueden elegir resolver cualquier problema, siempre y cuando el startup esté alineado con al menos un ODS.

**Campus Director en Universidad:** Miranda Erribarren Marin

---

## 2. Elegibilidad y Registro

### **Criterios de Elegibilidad**

1. **Estudiantes:**  
   - Deben estar matriculados en algún curso/ciclo/especialización universitaria (pregrado o posgrado) y tener al menos 18 años cumplidos al 28 de febrero de 2025.

2. **Equipos:**  
   - Cada equipo debe estar formado por **2 a 4 integrantes**.  
   - Los equipos pueden incluir estudiantes de diferentes facultades o incluso universidades, pero deben representar a una sola universidad, con al menos un miembro inscrito en ella.  
   - Todos los participantes deben estar registrados oficialmente en el formulario del equipo.

3. **Restricciones:**  
   - Los equipos se pueden inscribir en solo un programa OnCampus: representar a solo una universidad.

---

### **Proceso de Registro**

1. Completar el formulario de registro oficial antes del **25 de febrero**.  
2. Confirmación de registro: recibirán un correo electrónico con detalles adicionales para la participación.

---

## 3. Proceso del Concurso

### **Fase Local - OnCampus (Enero - Febrero)**

#### **1. Fase de Inscripción**  
Los equipos deberán registrarse antes del **25 de febrero de 2025**.

#### **2. Fase de Desarrollo**  
- Los equipos participarán en sesiones de mentoría y talleres durante febrero. La asistencia es obligatoria por al menos un miembro del equipo.  
- Tendrán la oportunidad de recibir retroalimentación directa para mejorar su modelo de negocio, impacto y pitch.  
- Tendrán acceso a **Hultie**, el WhatsApp agent que brinda retroalimentación y resuelve dudas frecuentes.

#### **3. Evento Final OnCampus**  
- **Fecha y formato:** 15/03/25, presencial o virtual.  
- Cada equipo tendrá 4 minutos para presentar su pitch ante un panel de jueces expertos, seguido de 4 minutos de preguntas y respuestas.  
- **Certificados:** Los equipos que completen su pitch y de los que al menos 1 integrante haya asistido a las mentorías y talleres recibirán el certificado oficial de participación.  
- **Selección:** El equipo ganador será anunciado al final del evento final del OnCampus y representará a la universidad en la **National Competition**.

---

### **National Competition (Mayo)**

Los ganadores de cada OnCampus competirán en la **National Competition**. Los equipos presentarán un pitch en inglés ante un panel de expertos. Serán 4 minutos de pitch y 4 de preguntas. El ganador nacional avanzará al **Hult Prize Digital Incubator**.

---

### **Digital Incubator (Junio - Julio)**

Hasta 60 startups globales serán seleccionadas para participar en este programa intensivo, donde recibirán:

- Mentorías personalizadas con expertos globales.  
- Talleres sobre diseño de impacto, validación de mercado, desarrollo de MVP y estrategias de escalamiento.  
- Recursos educativos digitales y acceso a herramientas para desarrollar su empresa.

Al final del programa, las startups presentarán sus avances y los mejores equipos avanzarán al **Hult Prize Global Accelerator**.

---

### **Global Accelerator (Agosto)**

Hasta 25 startups participarán en un programa presencial de un mes en Ashridge House, Reino Unido. Las startups recibirán:

- Acceso a inversionistas, socios potenciales y líderes de la industria.  
- Múltiples Demo Days para recibir retroalimentación directa y preparar sus propuestas finales.

Al final del programa, las startups presentarán sus avances y las seis mejores startups serán seleccionadas para competir en la **Global Final**.

---

### **Global Final (Septiembre)**

Los seis equipos finalistas presentarán sus proyectos ante un panel de jueces internacionales el **5 de septiembre en Londres**. El equipo ganador recibirá el gran premio de **$1 millón USD** para escalar su startup.

---

## 4. Criterios de Evaluación

Los jueces evaluarán las propuestas con base en cuatro categorías principales:

### **1. Equipo**

- **Organización:** Roles definidos y claridad en responsabilidades.  
- **Colaboración:** Capacidad de trabajar juntos para desarrollar y presentar la idea.  
- **Experiencia y habilidades:** Competencias complementarias y alineadas con el proyecto.

**Escala:**  
- 1: Desorganizado, roles poco claros y/o falta de experiencia.  
- 3: Equipo promedio con roles básicos y comunicación moderada.  
- 5: Equipo cohesionado, roles claros, excelente comunicación y complementariedad.

---

### **2. Idea**

- **Identificación del problema:** Comprensión clara del problema social o ambiental que se busca resolver.  
- **Solución innovadora:** Idea creativa y viable.  
- **Validación:** Existencia de pruebas, retroalimentación de usuarios o prototipos iniciales.

**Escala:**  
- 1: Problema poco definido, solución vaga y sin evidencia.  
- 3: Problema claro, solución viable, con validación básica.  
- 5: Solución probada, con retroalimentación robusta y potencial de impacto.

---

### **3. Impacto**

- **Alineación:** Relación directa con al menos un ODS.  
- **Medición:** Definición clara de KPIs y métricas de impacto.  
- **Escalabilidad:** Potencial para expandir el impacto social a medida que crecen los ingresos.

**Escala:**  
- 1: Sin alineación con ODS ni métricas claras.  
- 3: Alineación con ODS y métricas iniciales.  
- 5: Impacto claro y escalable, con KPIs bien definidos.

---

### **4. Viabilidad del Negocio**

- **Modelo de negocio:** Estructura clara, sostenible y realista.  
- **Economía unitaria:** Comprensión de los costos, ingresos y rentabilidad.  
- **Ventaja competitiva:** Elementos diferenciadores frente a otras soluciones.

**Escala:**  
- 1: Modelo débil y poco realista.  
- 3: Modelo claro pero con áreas por fortalecer.  
- 5: Modelo sólido, escalable y con alto potencial disruptivo.

---

## 5. Mentorías y Recursos

- **Mentorías personalizadas:** Disponibles en febrero, con acceso a expertos locales e internacionales.  
- **Talleres exclusivos:** Sesiones en vivo sobre creación de modelos de negocio, habilidades para pitching, etc.  
- **Ponencias Nacionales:** Oportunidad de asistir a talleres de expertos reconocidos globalmente del ecosistema de innovación y startups.  
- **Material grabado:** Acceso a grabaciones de talleres y recursos adicionales durante todo el programa.

---

## 6. Premios

1. **Primer Lugar en el OnCampus Program:**  
   - Avance directo a la Competencia Nacional Hult Prize.  
   - Certificado de excelencia.

2. **Otros premios:**  
   - Certificados para todos los participantes que completen el programa.

---

## 7. Términos y Condiciones

1. La participación implica la aceptación de estas bases y del **Código de Conducta Hult Prize**.  
2. Los participantes son responsables de los costos asociados con su participación (transporte, alimentación, etc.).  
3. Al menos un miembro del equipo ganador debe asistir presencialmente al evento final.  
4. Cualquier incumplimiento de las normas puede resultar en la descalificación inmediata.

---

## Contacto

- **Correo:** irribarrenmarinmiranda@gmail.com  
- **WhatsApp de contacto:** 930380900
"""
retriever_UNI = """ 
# Bases del Concurso OnCampus Hult Prize  
### “Universidad Nacional de Ingeniería”

## 1. Introducción
Hult Prize es la competencia estudiantil más grande del mundo que inspira a jóvenes emprendedores a resolver los mayores desafíos globales mediante startups sostenibles. Se conoce como el “Premio Nobel de los Estudiantes”. Desde su fundación, más de 1 millón de estudiantes han participado, buscando generar impacto positivo alineado con los **Objetivos de Desarrollo Sostenible (ODS)** de la ONU.

Para esta edición (2025), el desafío es **“Unlimited”**: Los equipos pueden elegir resolver cualquier problema, siempre y cuando la startup esté alineada con al menos un ODS.

**Campus Director en “Universidad”**: Jose Saul Canturin Cuyubamba

---

## 2. Elegibilidad y Registro

### Criterios de Elegibilidad:

1. **Estudiantes**:
   - Deben estar matriculados en algún curso/ciclo/especialización universitaria (pregrado o posgrado) y tener al menos 18 años cumplidos al 28 de febrero de 2025.

2. **Equipos**:
   - Cada equipo debe estar formado por **2 a 4 integrantes**.
   - Los equipos pueden incluir estudiantes de diferentes facultades o incluso universidades, pero deben representar a una sola universidad, con al menos un miembro inscrito en ella.
   - Todos los participantes deben estar registrados oficialmente en el formulario del equipo.

3. **Restricciones**:
   - Los equipos se pueden inscribir en solo un programa OnCampus: representar a solo una universidad.

### Proceso de Registro:

1. Completar el formulario oficial de registros antes del **02 de febrero**:  
   [Formulario de registro](https://docs.google.com/forms/d/e/1FAIpQLSc-atixFEYc0GtSTcoXhU9tDSEOxg6dbZEL3kdhkQ6G2eZ_7Q/viewform)

2. Confirmación de registro: recibirán un correo electrónico con detalles adicionales para la participación.

---

## 3. Proceso del Concurso

### Fase Local - OnCampus (Enero - Febrero)

1. **Fase de Inscripción**:
   - Los equipos deberán registrarse antes del **02 de febrero**.

2. **Fase de Desarrollo**:
   - Los equipos participarán en sesiones de mentoría y talleres durante enero. La asistencia es obligatoria por al menos un miembro del equipo.
   - Tendrán la oportunidad de recibir retroalimentación directa para mejorar su modelo de negocio, impacto y pitch.

3. **Evento Final OnCampus**:
   - **Fecha y formato**: 27 de febrero presencial.
   - Cada equipo tendrá **4 minutos** para presentar su pitch ante un panel de jueces expertos, seguido de **4 minutos** de preguntas y respuestas.
   - **Certificados**: Los equipos que completen su pitch y de los que al menos un integrante haya asistido a las mentorías y talleres recibirán el certificado oficial de participación.
   - **Selección**: El equipo ganador será anunciado al final del evento final del OnCampus y representará a la universidad en la **National Competition**.

---

### National Competition (Mayo)
Los ganadores de cada OnCampus competirán en la **National Competition**. Los equipos presentarán un pitch en inglés ante un panel de expertos. Serán **4 minutos** de pitch y **4 minutos** de preguntas. El ganador nacional avanzará al **Hult Prize Digital Incubator**.

---

### Digital Incubator (Junio - Julio)
Hasta **60 startups globales** serán seleccionadas para participar en este programa intensivo, donde recibirán:

- **Mentorías personalizadas** con expertos globales.
- **Talleres** sobre diseño de impacto, validación de mercado, desarrollo de MVP y estrategias de escalamiento.
- **Recursos educativos digitales** y acceso a herramientas para desarrollar su empresa.

Al final del programa, las startups presentarán sus avances y los mejores equipos avanzarán al **Hult Prize Global Accelerator**.

---

### Global Accelerator (Agosto)
Hasta **25 startups** participarán en un programa presencial de un mes en **Ashridge House, Reino Unido**. Las startups recibirán:

- Acceso a inversionistas, socios potenciales y líderes de la industria.
- Múltiples **Demo Days** para recibir retroalimentación directa y preparar sus propuestas finales.

Al final del programa, las startups presentarán sus avances y las seis mejores startups serán seleccionadas para competir en la **Global Final**.

---

### Global Final (Septiembre)
Los seis equipos finalistas presentarán sus proyectos ante un panel de jueces internacionales el **5 de septiembre en Londres**. El equipo ganador recibirá el gran premio de **$1 millón USD** para escalar su startup.

---

## 4. Criterios de Evaluación

Los jueces evaluarán las propuestas con base en cuatro categorías principales:

1. **Equipo**:
   - **Organización**: Roles definidos y claridad en responsabilidades.
   - **Colaboración**: Capacidad de trabajar juntos para desarrollar y presentar la idea.
   - **Experiencia y habilidades**: Competencias complementarias y alineadas con el proyecto.

2. **Idea**:
   - **Identificación del problema**: Comprensión clara del problema social o ambiental que se busca resolver.
   - **Solución innovadora**: Idea creativa y viable.

3. **Impacto**:
   - **Alineación**: Relación directa con al menos un ODS.
   - **Medición**: Definición clara de KPIs y métricas de impacto.

4. **Viabilidad del Negocio**:
   - **Modelo de negocio**: Estructura clara, sostenible y realista.
   - **Ventaja competitiva**: Elementos diferenciadores frente a otras soluciones.

---

## 5. Mentorías y Recursos

- **Mentorías personalizadas**: Disponibles en enero, con acceso a expertos locales e internacionales.
- **Talleres exclusivos**: Sesiones en vivo sobre creación de modelos de negocio, habilidades para pitching, etc.
- **Ponencias Nacionales**: Oportunidad de asistir a talleres de expertos reconocidos globalmente del ecosistema de innovación y startups.
- **Material grabado**: Acceso a grabaciones de talleres y recursos adicionales durante todo el programa.

---

## 6. Premios

1. **Primer Lugar en el OnCampus Program**:
   - Avance directo a la **Competencia Nacional Hult Prize**.
   - Certificado de excelencia.

2. **Otros premios**:
   - Certificados para todos los participantes que completen el programa.

---

## 7. Términos y Condiciones

1. La participación implica la aceptación de estas bases y del **Código de Conducta Hult Prize**.
2. Los participantes son responsables de los costos asociados con su participación (transporte, alimentación, etc.).
3. Al menos un miembro del equipo ganador debe asistir presencialmente al evento final.
4. Cualquier incumplimiento de las normas puede resultar en la descalificación inmediata.

---

**Contacto**:  
Jose Canturin Cuyubamba  
[jose.canturin.c@uni.pe](mailto:jose.canturin.c@uni.pe)  
+51 972 893 391



"""
retriever_USIL = """
# Bases del Concurso OnCampus Hult Prize
**Universidad San Ignacio de Loyola**

## 1. Introducción
Hult Prize es la competencia estudiantil más grande del mundo que inspira a jóvenes emprendedores a resolver los mayores desafíos globales mediante startups sostenibles. Se conoce como el “Premio Nobel de los Estudiantes”. Desde su fundación más de 1 millón de estudiantes han participado, buscando generar impacto positivo alineado con los Objetivos de Desarrollo Sostenible (ODS) de la ONU.

Para esta edición (2025) el desafío es **“Unlimited”**: Los equipos pueden elegir resolver cualquier problema, siempre y cuando la startup esté alineada con al menos un ODS.

**Campus Director en Universidad San Ignacio de Loyola**: Rocio Tamara Laura Villanueva

---

## 2. Elegibilidad y Registro

### Criterios de Elegibilidad:

1. **Estudiantes:**
   - Deben estar matriculados en algún curso/ciclo/especialización universitaria (pregrado o posgrado) y tener al menos 18 años cumplidos al 28 de febrero de 2025.

2. **Equipos:**
   - Cada equipo debe estar formado por **2 a 4 integrantes**.
   - Los equipos pueden incluir estudiantes de diferentes facultades o incluso universidades, pero deben representar a una sola universidad, con al menos un miembro inscrito en ella.
   - Todos los participantes deben estar registrados oficialmente en el formulario del equipo.

3. **Restricciones:**
   - Los equipos se pueden inscribir en solo un programa OnCampus: representar a solo una universidad.

### Proceso de Registro:

1. Completar el formulario de registro oficial antes del **31 de enero**.
2. Confirmación de registro: recibirán un correo electrónico con detalles adicionales para la participación.

---

## 3. Proceso del Concurso

### Fase Local - OnCampus (Enero - Febrero)

1. **Fase de Inscripción:**
   - Los equipos deberán registrarse antes del 31 de enero de 2025.

2. **Fase de Desarrollo:**
   - Los equipos participarán en sesiones de mentoría y talleres durante febrero. La asistencia es obligatoria por al menos un miembro del equipo.
   - Tendrán la oportunidad de recibir retroalimentación directa para mejorar su modelo de negocio, impacto y pitch.
   - Tendrán acceso a Hultie, el WhatsApp que brinda retroalimentación y resuelve dudas frecuentes.

3. **Evento Final OnCampus:**
   - **Fecha y formato:** 28 de febrero (fecha tentativa), presencial.
   - Cada equipo tendrá **4 minutos** para presentar su pitch ante un panel de jueces expertos, seguido de **4 minutos de preguntas y respuestas**.
   - **Certificados:** Los equipos que completen su pitch y de los que al menos 1 integrante haya asistido a las mentorías y talleres recibirán el certificado oficial de participación.
   - **Selección:** El equipo ganador será anunciado al final del evento final del OnCampus y representará a la universidad en la **National Competition**.

---

### National Competition (Mayo)

Los ganadores de cada OnCampus competirán en la **National Competition**. Los equipos presentarán un pitch en inglés ante un panel de expertos. Serán **4 minutos de pitch y 4 de preguntas**. El ganador nacional avanzará al **Hult Prize Digital Incubator**.

---

### Digital Incubator (Junio - Julio)

Hasta **60 startups globales** serán seleccionadas para participar en este programa intensivo, donde recibirán:

- **Mentorías personalizadas** con expertos globales.
- **Talleres exclusivos** sobre diseño de impacto, validación de mercado, desarrollo de MVP y estrategias de escalamiento.
- **Recursos educativos digitales** y acceso a herramientas para desarrollar su empresa.

Al final del programa, las startups presentarán sus avances y los mejores equipos avanzarán al **Hult Prize Global Accelerator**.

---

### Global Accelerator (Agosto)

Hasta **25 startups** participarán en un programa presencial de un mes en **Ashridge House, Reino Unido**. Las startups recibirán:

- **Acceso a inversionistas**, socios potenciales y líderes de la industria.
- **Múltiples Demo Days** para recibir retroalimentación directa y preparar sus propuestas finales.

Al final del programa, las startups presentarán sus avances y las seis mejores startups serán seleccionadas para competir en la **Global Final**.

---

### Global Final (Septiembre)

Los seis equipos finalistas presentarán sus proyectos ante un panel de jueces internacionales el **5 de septiembre en Londres**. El equipo ganador recibirá el gran premio de **$1 millón USD** para escalar su startup.

---

## 4. Criterios de Evaluación

Los jueces evaluarán las propuestas con base en cuatro categorías principales:

1. **Equipo:**
   - **Organización:** Roles definidos y claridad en responsabilidades.
   - **Colaboración:** Capacidad de trabajar juntos para desarrollar y presentar la idea.
   - **Experiencia y habilidades:** Competencias complementarias y alineadas con el proyecto.
   - **Escala:**
     - **1:** Desorganizado, roles poco claros y/o falta de experiencia.
     - **3:** Equipo promedio con roles básicos y comunicación moderada.
     - **5:** Equipo cohesionado, roles claros, excelente comunicación y complementariedad.

2. **Idea:**
   - **Identificación del problema:** Comprensión clara del problema social o ambiental que se busca resolver.
   - **Solución innovadora:** Idea creativa y viable.
   - **Validación:** Existencia de pruebas, retroalimentación de usuarios o prototipos iniciales.
   - **Escala:**
     - **1:** Problema poco definido, solución vaga y sin evidencia.
     - **3:** Problema claro, solución viable, con validación básica.
     - **5:** Solución probada, con retroalimentación robusta y potencial de impacto.

3. **Impacto:**
   - **Alineación:** Relación directa con al menos un ODS.
   - **Medición:** Definición clara de KPIs y métricas de impacto.
   - **Escalabilidad:** Potencial para expandir el impacto social a medida que crecen los ingresos.
   - **Escala:**
     - **1:** Sin alineación con ODS ni métricas claras.
     - **3:** Alineación con ODS y métricas iniciales.
     - **5:** Impacto claro y escalable, con KPIs bien definidos.

4. **Viabilidad del Negocio:**
   - **Modelo de negocio:** Estructura clara, sostenible y realista.
   - **Economía unitaria:** Comprensión de los costos, ingresos y rentabilidad.
   - **Ventaja competitiva:** Elementos diferenciadores frente a otras soluciones.
   - **Escala:**
     - **1:** Modelo débil y poco realista.
     - **3:** Modelo claro pero con áreas por fortalecer.
     - **5:** Modelo sólido, escalable y con alto potencial disruptivo.

---

## 5. Mentorías y Recursos

- **Mentorías personalizadas:** Disponibles en febrero con acceso a expertos locales e internacionales.
- **Talleres exclusivos:** Sesiones en vivo sobre creación de modelos de negocio, habilidades para pitching, etc.
- **Ponencias Nacionales:** Oportunidad de asistir a talleres de expertos reconocidos globalmente del ecosistema de innovación y startups.
- **Material grabado:** Acceso a grabaciones de talleres y recursos adicionales durante todo el programa.

---

## 6. Premios

1. **Primer Lugar en el OnCampus Program:**
   - Avance directo a la Competencia Nacional Hult Prize.
   - Certificado de excelencia.
   - [Otros premios específicos si aplica, por ejemplo: becas, mentorías adicionales].

---

## 7. Términos y Condiciones

1. La participación implica la aceptación de estas bases y del **Código de Conducta Hult Prize**.
2. Los participantes son responsables de los costos asociados con su participación (transporte, alimentación, etc.).
3. Al menos un miembro del equipo ganador debe asistir presencialmente al evento final.
4. Cualquier incumplimiento de las normas puede resultar en la descalificación inmediata.

---

**Contacto:**  
**Correo:** hultusilprize@gmail.com  
**Campus Director:** 960132020
"""
retriever_UDEP = """
# Bases del Concurso OnCampus Hult Prize  
### Universidad de Piura

## 1. Introducción
Hult Prize es la competencia estudiantil más grande del mundo que inspira a jóvenes emprendedores a resolver los mayores desafíos globales mediante startups sostenibles. Se conoce como el “Premio Nobel de los Estudiantes”. Desde su fundación, más de 1 millón de estudiantes han participado, buscando generar impacto positivo alineado con los **Objetivos de Desarrollo Sostenible (ODS)** de la ONU.

Para esta edición (2025), el desafío es **“Unlimited”**: Los equipos pueden elegir resolver cualquier problema, siempre y cuando la startup esté alineada con al menos un ODS.

**Campus Director en UDEP**: Emili Rodriguez

---

## 2. Elegibilidad y Registro

### Criterios de Elegibilidad:

1. **Estudiantes**:
   - Deben estar matriculados en algún curso/ciclo/especialización universitaria (pregrado o posgrado) y tener al menos 18 años cumplidos al 28 de febrero de 2025.

2. **Equipos**:
   - Cada equipo debe estar formado por **2 a 4 integrantes**.
   - Los equipos pueden incluir estudiantes de diferentes facultades o incluso universidades, pero deben representar a una sola universidad, con al menos un miembro inscrito en ella.
   - Todos los participantes deben estar registrados oficialmente en el formulario del equipo.

3. **Restricciones**:
   - Los equipos se pueden inscribir en solo un programa OnCampus: representar a solo una universidad.

### Proceso de Registro:

1. Completar el formulario de registro oficial antes del **31 de enero**.  
   [Formulario de registro](https://docs.google.com/forms)

2. Confirmación de registro: recibirán un correo electrónico con detalles adicionales para la participación.

---

## 3. Proceso del Concurso

### Fase Local - OnCampus (Enero - Febrero)

1. **Fase de Inscripción**:
   - Los equipos deberán registrarse antes del **31 de enero de 2025**.

2. **Fase de Desarrollo**:
   - Los equipos participarán en sesiones de mentoría y talleres durante enero. La asistencia es obligatoria por al menos un miembro del equipo.
   - Tendrán la oportunidad de recibir retroalimentación directa para mejorar su modelo de negocio, impacto y pitch.
   - Tendrán acceso a **Hultie**, el **wsp agent** que brinda retroalimentación y resuelve dudas frecuentes.

3. **Evento Final OnCampus**:
   - **Fecha y formato**: En programación, presencial.
   - Cada equipo tendrá **4 minutos** para presentar su pitch ante un panel de jueces expertos, seguido de **4 minutos** de preguntas y respuestas.
   - **Certificados**: Los equipos que completen su pitch y de los que al menos un integrante haya asistido a las mentorías y talleres recibirán el certificado oficial de participación.
   - **Selección**: El equipo ganador será anunciado al final del evento final del OnCampus y representará a la universidad en la **National Competition**.

---

### National Competition (Mayo)
Los ganadores de cada OnCampus competirán en la **National Competition**. Los equipos presentarán un pitch en inglés ante un panel de expertos. Serán **4 minutos** de pitch y **4 minutos** de preguntas. El ganador nacional avanzará al **Hult Prize Digital Incubator**.

---

### Digital Incubator (Junio - Julio)
Hasta **60 startups globales** serán seleccionadas para participar en este programa intensivo, donde recibirán:

- **Mentorías personalizadas** con expertos globales.
- **Talleres** sobre diseño de impacto, validación de mercado, desarrollo de MVP y estrategias de escalamiento.
- **Recursos educativos digitales** y acceso a herramientas para desarrollar su empresa.

Al final del programa, las startups presentarán sus avances y los mejores equipos avanzarán al **Hult Prize Global Accelerator**.

---

### Global Accelerator (Agosto)
Hasta **25 startups** participarán en un programa presencial de un mes en **Ashridge House, Reino Unido**. Las startups recibirán:

- Acceso a inversionistas, socios potenciales y líderes de la industria.
- Múltiples **Demo Days** para recibir retroalimentación directa y preparar sus propuestas finales.

Al final del programa, las startups presentarán sus avances y las seis mejores startups serán seleccionadas para competir en la **Global Final**.

---

### Global Final (Septiembre)
Los seis equipos finalistas presentarán sus proyectos ante un panel de jueces internacionales el **5 de septiembre en Londres**. El equipo ganador recibirá el gran premio de **$1 millón USD** para escalar su startup.

---

## 4. Criterios de Evaluación

Los jueces evaluarán las propuestas con base en cuatro categorías principales:

1. **Equipo**:
   - **Organización**: Roles definidos y claridad en responsabilidades.
   - **Colaboración**: Capacidad de trabajar juntos para desarrollar y presentar la idea.
   - **Experiencia y habilidades**: Competencias complementarias y alineadas con el proyecto.

2. **Idea**:
   - **Identificación del problema**: Comprensión clara del problema social o ambiental que se busca resolver.
   - **Solución innovadora**: Idea creativa y viable.
   - **Validación**: Existencia de pruebas, retroalimentación de usuarios o prototipos iniciales.

3. **Impacto**:
   - **Alineación**: Relación directa con al menos un ODS.
   - **Medición**: Definición clara de KPIs y métricas de impacto.

4. **Viabilidad del Negocio**:
   - **Modelo de negocio**: Estructura clara, sostenible y realista.
   - **Ventaja competitiva**: Elementos diferenciadores frente a otras soluciones.

---

## 5. Mentorías y Recursos

- **Mentorías personalizadas**: Disponibles en febrero, con acceso a expertos locales e internacionales.
- **Talleres exclusivos**: Sesiones en vivo sobre creación de modelos de negocio, habilidades para pitching, etc.
- **Ponencias Nacionales**: Oportunidad de asistir a talleres de expertos reconocidos globalmente del ecosistema de innovación y startups.
- **Material grabado**: Acceso a grabaciones de talleres y recursos adicionales durante todo el programa.

---

## 6. Premios

1. **Primer Lugar en el OnCampus Program**:
   - Avance directo a la **Competencia Nacional Hult Prize**.
   - Certificado de excelencia.

2. **Otros premios**:
   - Certificados para todos los participantes que completen el programa.

---

## 7. Términos y Condiciones

1. La participación implica la aceptación de estas bases y del **Código de Conducta Hult Prize**.
2. Los participantes son responsables de los costos asociados con su participación (transporte, alimentación, etc.).
3. Al menos un miembro del equipo ganador debe asistir presencialmente al evento final.
4. Cualquier incumplimiento de las normas puede resultar en la descalificación inmediata.

---

**Contacto**:  
Emili Rodriguez  
[emili.rodriguez@alum.udep.edu.pe](mailto:emili.rodriguez@alum.udep.edu.pe)  
+51 906 263 443
"""
retriever_UCSUR = """Todavía no se han publicado las bases de UCSUR, por favor contacta a su Campus Director, **Alejandra Vera**, quien se encarga del evento, para que las publique.
Sin embargo, aca esta la información general independiente de sede:
### National Competition (Mayo)
Los ganadores de cada OnCampus competirán en la **National Competition**. Los equipos presentarán un pitch en inglés ante un panel de expertos. Serán **4 minutos de pitch y 4 de preguntas**. El ganador nacional avanzará al **Hult Prize Digital Incubator**.

### Digital Incubator (Junio - Julio)
Hasta **60 startups globales** serán seleccionadas para participar en este programa intensivo, donde recibirán:
- **Mentorías personalizadas con expertos globales**.
- **Talleres sobre diseño de impacto, validación de mercado, desarrollo de MVP y estrategias de escalamiento**.
- **Recursos educativos digitales y acceso a herramientas para desarrollar su empresa**.

Al final del programa, las startups presentarán sus avances y los mejores equipos avanzarán al **Hult Prize Global Accelerator**.

### Global Accelerator (Agosto)
Hasta **25 startups** participarán en un programa presencial de un mes en **Ashridge House, Reino Unido**. Las startups recibirán:
- **Acceso a inversionistas, socios potenciales y líderes de la industria**.
- **Múltiples Demo Days** para recibir retroalimentación directa y preparar sus propuestas finales.

Al final del programa, las startups presentarán sus avances y las seis mejores startups serán seleccionadas para competir en la **Global Final**.

### Global Final (Septiembre)
Los seis equipos finalistas presentarán sus proyectos ante un panel de jueces internacionales el **5 de septiembre en Londres**. El equipo ganador recibirá el gran premio de **$1 millón USD** para escalar su startup.
"""
retriever_UARM = """
# Bases del Concurso OnCampus Hult Prize  
**Universidad Antonio Ruiz de Montoya**

## 1. Introducción

Hult Prize es la competencia estudiantil más grande del mundo que inspira a jóvenes emprendedores a resolver los mayores desafíos globales mediante startups sostenibles. Se conoce como el “Premio Nobel de los Estudiantes”. Desde su fundación más de 1 millón de estudiantes han participado, buscando generar impacto positivo alineado con los Objetivos de Desarrollo Sostenible (ODS) de la ONU.

Para esta edición (2025) el desafío es **“Unlimited”**: Los equipos pueden elegir resolver cualquier problema, siempre y cuando la startup esté alineada con al menos un ODS.

**Campus Director en Universidad Antonio Ruiz de Montoya**: Patme Estefanía Villavicencio Chavez

---

## 2. Elegibilidad y Registro

### Criterios de Elegibilidad:

1. **Estudiantes**:
   - Deben estar matriculados en algún curso/ciclo/especialización universitaria (pregrado o posgrado) y tener al menos 18 años cumplidos al 28 de febrero de 2025.

2. **Equipos**:
   - Cada equipo debe estar formado por **2 a 4 integrantes**.
   - Los equipos pueden incluir estudiantes de diferentes facultades o incluso universidades, pero deben representar a una sola universidad, con al menos un miembro inscrito en ella.
   - Todos los participantes deben estar registrados oficialmente en el formulario del equipo.

3. **Restricciones**:
   - Los equipos se pueden inscribir en solo un programa OnCampus: representar a solo una universidad.

### Proceso de Registro:

1. Completar el formulario de registro oficial antes del **25 de febrero**.
2. Confirmación de registro: recibirán un correo electrónico con detalles adicionales para la participación.

---

## 3. Proceso del Concurso

### Fase Local - OnCampus (Enero - Febrero)

1. **Fase de Inscripción**:
   - Los equipos deberán registrarse antes del 25 de febrero de 2025.

2. **Fase de Desarrollo**:
   - Los equipos participarán en sesiones de mentoría y talleres durante febrero. La asistencia es obligatoria por al menos un miembro del equipo.
   - Tendrán la oportunidad de recibir retroalimentación directa para mejorar su modelo de negocio, impacto y pitch.
   - Tendrán acceso a Hultie, el WhatsApp agente que brinda retroalimentación y resuelve dudas frecuentes.

3. **Evento Final OnCampus**:
   - **Fecha y formato**: [detallar], presencial.
   - Cada equipo tendrá **4 minutos** para presentar su pitch ante un panel de jueces expertos, seguido de **4 minutos de preguntas y respuestas**.
   - **Certificados**: Los equipos que completen su pitch y de los que al menos 1 integrante haya asistido a las mentorías y talleres recibirán el certificado oficial de participación.
   - **Selección**: El equipo ganador será anunciado al final del evento final del OnCampus y representará a la universidad en la **National Competition**.

---

### National Competition (Mayo)

Los ganadores de cada OnCampus competirán en la **National Competition**. Los equipos presentarán un pitch en inglés ante un panel de expertos. Serán **4 minutos de pitch y 4 de preguntas**. El ganador nacional avanzará al **Hult Prize Digital Incubator**.

---

### Digital Incubator (Junio - Julio)

Hasta **60 startups globales** serán seleccionadas para participar en este programa intensivo, donde recibirán:

- **Mentorías personalizadas** con expertos globales.
- **Talleres** sobre diseño de impacto, validación de mercado, desarrollo de MVP y estrategias de escalamiento.
- **Recursos educativos digitales** y acceso a herramientas para desarrollar su empresa.

Al final del programa, las startups presentarán sus avances y los mejores equipos avanzarán al **Hult Prize Global Accelerator**.

---

### Global Accelerator (Agosto)

Hasta **25 startups** participarán en un programa presencial de un mes en **Ashridge House, Reino Unido**. Las startups recibirán:

- **Acceso a inversionistas**, socios potenciales y líderes de la industria.
- **Múltiples Demo Days** para recibir retroalimentación directa y preparar sus propuestas finales.

Al final del programa, las startups presentarán sus avances y las seis mejores startups serán seleccionadas para competir en la **Global Final**.

---

### Global Final (Septiembre)

Los seis equipos finalistas presentarán sus proyectos ante un panel de jueces internacionales el **5 de septiembre en Londres**. El equipo ganador recibirá el gran premio de **$1 millón USD** para escalar su startup.

---

## 4. Criterios de Evaluación

Los jueces evaluarán las propuestas con base en cuatro categorías principales:

1. **Equipo**:
   - **Organización**: Roles definidos y claridad en responsabilidades.
   - **Colaboración**: Capacidad de trabajar juntos para desarrollar y presentar la idea.
   - **Experiencia y habilidades**: Competencias complementarias y alineadas con el proyecto.
   - **Escala**:
     - **1**: Desorganizado, roles poco claros y/o falta de experiencia.
     - **3**: Equipo promedio con roles básicos y comunicación moderada.
     - **5**: Equipo cohesionado, roles claros, excelente comunicación y complementariedad.

2. **Idea**:
   - **Identificación del problema**: Comprensión clara del problema social o ambiental que se busca resolver.
   - **Solución innovadora**: Idea creativa y viable.
   - **Validación**: Existencia de pruebas, retroalimentación de usuarios o prototipos iniciales.
   - **Escala**:
     - **1**: Problema poco definido, solución vaga y sin evidencia.
     - **3**: Problema claro, solución viable, con validación básica.
     - **5**: Solución probada, con retroalimentación robusta y potencial de impacto.

3. **Impacto**:
   - **Alineación**: Relación directa con al menos un ODS.
   - **Medición**: Definición clara de KPIs y métricas de impacto.
   - **Escalabilidad**: Potencial para expandir el impacto social a medida que crecen los ingresos.
   - **Escala**:
     - **1**: Sin alineación con ODS ni métricas claras.
     - **3**: Alineación con ODS y métricas iniciales.
     - **5**: Impacto claro y escalable, con KPIs bien definidos.

4. **Viabilidad del Negocio**:
   - **Modelo de negocio**: Estructura clara, sostenible y realista.
   - **Economía unitaria**: Comprensión de los costos, ingresos y rentabilidad.
   - **Ventaja competitiva**: Elemento
"""
retriever_ESAN = """Todavía no se han publicado las bases de ESAN, por favor contacta a su Campus Director, **Diego Zorrilla**, quien se encarga del evento, para que las publique.
Sin embargo, aca esta la información general independiente de sede:
### National Competition (Mayo)
Los ganadores de cada OnCampus competirán en la **National Competition**. Los equipos presentarán un pitch en inglés ante un panel de expertos. Serán **4 minutos de pitch y 4 de preguntas**. El ganador nacional avanzará al **Hult Prize Digital Incubator**.

### Digital Incubator (Junio - Julio)
Hasta **60 startups globales** serán seleccionadas para participar en este programa intensivo, donde recibirán:
- **Mentorías personalizadas con expertos globales**.
- **Talleres sobre diseño de impacto, validación de mercado, desarrollo de MVP y estrategias de escalamiento**.
- **Recursos educativos digitales y acceso a herramientas para desarrollar su empresa**.

Al final del programa, las startups presentarán sus avances y los mejores equipos avanzarán al **Hult Prize Global Accelerator**.

### Global Accelerator (Agosto)
Hasta **25 startups** participarán en un programa presencial de un mes en **Ashridge House, Reino Unido**. Las startups recibirán:
- **Acceso a inversionistas, socios potenciales y líderes de la industria**.
- **Múltiples Demo Days** para recibir retroalimentación directa y preparar sus propuestas finales.

Al final del programa, las startups presentarán sus avances y las seis mejores startups serán seleccionadas para competir en la **Global Final**.

### Global Final (Septiembre)
Los seis equipos finalistas presentarán sus proyectos ante un panel de jueces internacionales el **5 de septiembre en Londres**. El equipo ganador recibirá el gran premio de **$1 millón USD** para escalar su startup.

"""
retriever_UCAL = """
# Bases del Concurso OnCampus Hult Prize
**Universidad de Ciencias y Artes de América Latina (UCAL)**

## 1. Introducción
Hult Prize es la competencia estudiantil más grande del mundo que inspira a jóvenes emprendedores a resolver los mayores desafíos globales mediante startups sostenibles. Se conoce como el “Premio Nobel de los Estudiantes”. Desde su fundación más de 1 millón de estudiantes han participado, buscando generar impacto positivo alineado con los Objetivos de Desarrollo Sostenible (ODS) de la ONU.

Para esta edición (2025) el desafío es **“Unlimited”**: Los equipos pueden elegir resolver cualquier problema, siempre y cuando la startup esté alineada con al menos un ODS.

**Campus Director en UCAL**: Gustavo Cosme

---

## 2. Elegibilidad y Registro

### Criterios de Elegibilidad:
1. **Estudiantes**:
   - Deben estar matriculados en algún curso/ciclo/especialización universitaria (pregrado o posgrado) y tener al menos 18 años cumplidos al 28 de febrero de 2025.

2. **Equipos**:
   - Cada equipo debe estar formado por **2 a 4 integrantes**.
   - Los equipos pueden incluir estudiantes de diferentes facultades o incluso universidades, pero deben representar a una sola universidad, con al menos un miembro inscrito en ella.
   - Todos los participantes deben estar registrados oficialmente en el formulario del equipo.

3. **Restricciones**:
   - Los equipos se pueden inscribir en solo un programa OnCampus: representar a solo una universidad.

### Proceso de Registro:
1. **Completar el formulario de registro oficial antes del 25 de febrero**.
2. **Confirmación de registro**: recibirán un correo electrónico con detalles adicionales para la participación.

---

## 3. Proceso del Concurso

### Fase Local - OnCampus (Enero - Febrero)
1. **Fase de Inscripción**:
   - Los equipos deberán registrarse antes del 31 de enero de 2025.

2. **Fase de Desarrollo**:
   - Los equipos participarán en sesiones de mentoría y talleres durante febrero. La asistencia es obligatoria por al menos un miembro del equipo.
   - Tendrán la oportunidad de recibir retroalimentación directa para mejorar su modelo de negocio, impacto y pitch.
   - Tendrán acceso a **Hultie**, el wsp agente que brinda retroalimentación y resuelve dudas frecuentes.

3. **Evento Final OnCampus**:
   - **Fecha y formato**: 25 de febrero, virtual.
   - Cada equipo tendrá **5 minutos para presentar su pitch** ante un panel de jueces expertos, seguido de **5 minutos de preguntas y respuestas**.
   - **Certificados**: Los equipos que completen su pitch y de los que al menos un integrante haya asistido a las mentorías y talleres recibirán el certificado oficial de participación.
   - **Selección**: El equipo ganador será anunciado al final del evento final del OnCampus y representará a la UCAL en la **National Competition**.

### National Competition (Mayo)
Los ganadores de cada OnCampus competirán en la **National Competition**. Los equipos presentarán un pitch en inglés ante un panel de expertos. Serán **4 minutos de pitch y 4 de preguntas**. El ganador nacional avanzará al **Hult Prize Digital Incubator**.

### Digital Incubator (Junio - Julio)
Hasta **60 startups globales** serán seleccionadas para participar en este programa intensivo, donde recibirán:
- **Mentorías personalizadas con expertos globales**.
- **Talleres sobre diseño de impacto, validación de mercado, desarrollo de MVP y estrategias de escalamiento**.
- **Recursos educativos digitales y acceso a herramientas para desarrollar su empresa**.

Al final del programa, las startups presentarán sus avances y los mejores equipos avanzarán al **Hult Prize Global Accelerator**.

### Global Accelerator (Agosto)
Hasta **25 startups** participarán en un programa presencial de un mes en **Ashridge House, Reino Unido**. Las startups recibirán:
- **Acceso a inversionistas, socios potenciales y líderes de la industria**.
- **Múltiples Demo Days** para recibir retroalimentación directa y preparar sus propuestas finales.

Al final del programa, las startups presentarán sus avances y las seis mejores startups serán seleccionadas para competir en la **Global Final**.

### Global Final (Septiembre)
Los seis equipos finalistas presentarán sus proyectos ante un panel de jueces internacionales el **5 de septiembre en Londres**. El equipo ganador recibirá el gran premio de **$1 millón USD** para escalar su startup.

---

## 4. Criterios de Evaluación

Los jueces evaluarán las propuestas con base en cuatro categorías principales:

1. **Equipo**:
   - **Organización**: Roles definidos y claridad en responsabilidades.
   - **Colaboración**: Capacidad de trabajar juntos para desarrollar y presentar la idea.
   - **Experiencia y habilidades**: Competencias complementarias y alineadas con el proyecto.
   - **Escala**:
     - **1**: Desorganizado, roles poco claros y/o falta de experiencia.
     - **3**: Equipo promedio con roles básicos y comunicación moderada.
     - **5**: Equipo cohesionado, roles claros, excelente comunicación y complementariedad.

2. **Idea**:
   - **Identificación del problema**: Comprensión clara del problema social o ambiental que se busca resolver.
   - **Solución innovadora**: Idea creativa y viable.
   - **Validación**: Existencia de pruebas, retroalimentación de usuarios o prototipos iniciales.
   - **Escala**:
     - **1**: Problema poco definido, solución vaga y sin evidencia.
     - **3**: Problema claro, solución viable, con validación básica.
     - **5**: Solución probada, con retroalimentación robusta y potencial de impacto.

3. **Impacto**:
   - **Alineación**: Relación directa con al menos un ODS.
   - **Medición**: Definición clara de KPIs y métricas de impacto.
   - **Escalabilidad**: Potencial para expandir el impacto social a medida que crecen los ingresos.
   - **Escala**:
     - **1**: Sin alineación con ODS ni métricas claras.
     - **3**: Alineación con ODS y métricas iniciales.
     - **5**: Impacto claro y escalable, con KPIs bien definidos.

4. **Viabilidad del Negocio**:
   - **Modelo de negocio**: Estructura clara, sostenible y realista.
   - **Economía unitaria**: Comprensión de los costos, ingresos y rentabilidad.
   - **Ventaja competitiva**: Elementos diferenciadores frente a otras soluciones.
   - **Escala**:
     - **1**: Modelo débil y poco realista.
     - **3**: Modelo claro pero con áreas por fortalecer.
     - **5**: Modelo sólido, escalable y con alto potencial disruptivo.

---

## 5. Mentorías y Recursos
- **Mentorías personalizadas**: Disponibles en enero, con acceso a expertos locales e internacionales.
- **Talleres exclusivos**: Sesiones en vivo sobre creación de modelos de negocio, habilidades para pitching, etc.
- **Ponencias Nacionales**: Oportunidad de asistir a talleres de expertos reconocidos globalmente del ecosistema de innovación y startups.
- **Material grabado**: Acceso a grabaciones de talleres y recursos adicionales durante todo el programa.

---

## 7. Términos y Condiciones
1. La participación implica la aceptación de estas bases y del **Código de Conducta Hult Prize**.
2. Los participantes son responsables de los costos asociados con su participación (transporte, alimentación, etc.).
3. Al menos un miembro del equipo ganador debe asistir presencialmente al evento final.
4. Cualquier incumplimiento de las normas puede resultar en la descalificación inmediata.

**Contacto**:
- **Correo**: gmcosmev@crear.ucal.edu.pe
- **Teléfono**: +51932295930
"""
retriever_UTEC = """Todavía no se han publicado las bases de UTEC, por favor contacta a su Campus Director, **Gianfranco Dávila**, quien se encarga del evento, para que las publique.
Sin embargo, aca esta la información general independiente de sede:
### National Competition (Mayo)
Los ganadores de cada OnCampus competirán en la **National Competition**. Los equipos presentarán un pitch en inglés ante un panel de expertos. Serán **4 minutos de pitch y 4 de preguntas**. El ganador nacional avanzará al **Hult Prize Digital Incubator**.

### Digital Incubator (Junio - Julio)
Hasta **60 startups globales** serán seleccionadas para participar en este programa intensivo, donde recibirán:
- **Mentorías personalizadas con expertos globales**.
- **Talleres sobre diseño de impacto, validación de mercado, desarrollo de MVP y estrategias de escalamiento**.
- **Recursos educativos digitales y acceso a herramientas para desarrollar su empresa**.

Al final del programa, las startups presentarán sus avances y los mejores equipos avanzarán al **Hult Prize Global Accelerator**.

### Global Accelerator (Agosto)
Hasta **25 startups** participarán en un programa presencial de un mes en **Ashridge House, Reino Unido**. Las startups recibirán:
- **Acceso a inversionistas, socios potenciales y líderes de la industria**.
- **Múltiples Demo Days** para recibir retroalimentación directa y preparar sus propuestas finales.

Al final del programa, las startups presentarán sus avances y las seis mejores startups serán seleccionadas para competir en la **Global Final**.

### Global Final (Septiembre)
Los seis equipos finalistas presentarán sus proyectos ante un panel de jueces internacionales el **5 de septiembre en Londres**. El equipo ganador recibirá el gran premio de **$1 millón USD** para escalar su startup.

"""
retriever_PUCP = """
# Bases del Concurso OnCampus Hult Prize  
### Pontificia Universidad Católica del Perú

## 1. Introducción
Hult Prize es la competencia estudiantil más grande del mundo que inspira a jóvenes emprendedores a resolver los mayores desafíos globales mediante startups sostenibles. Se conoce como el “Premio Nobel de los Estudiantes”. Desde su fundación, más de 1 millón de estudiantes han participado, buscando generar impacto positivo alineado con los **Objetivos de Desarrollo Sostenible (ODS)** de la ONU.

Para esta edición (2025), el desafío es **“Unlimited”**: Los equipos pueden elegir resolver cualquier problema, siempre y cuando la startup esté alineada con al menos un ODS.

**Campus Director en “Universidad”**: Mehll Nayheli Mireya Rojas Ponce

---

## 2. Elegibilidad y Registro

### Criterios de Elegibilidad:

1. **Estudiantes**:
   - Deben estar matriculados en algún curso/ciclo/especialización universitaria (pregrado o posgrado) y tener al menos 18 años cumplidos al 28 de febrero de 2025.

2. **Equipos**:
   - Cada equipo debe estar formado por **2 a 4 integrantes**.
   - Los equipos pueden incluir estudiantes de diferentes facultades o incluso universidades, pero deben representar a una sola universidad, con al menos un miembro inscrito en ella.
   - Todos los participantes deben estar registrados oficialmente en el formulario del equipo.

3. **Restricciones**:
   - Los equipos se pueden inscribir en solo un programa OnCampus: representar a solo una universidad.

### Proceso de Registro:

1. Completar el formulario de registro oficial antes del **31 de enero**.  
   [Formulario de registro oficial](https://docs.google.com/forms)

2. Confirmación de registro: recibirán un correo electrónico con detalles adicionales para la participación.

---

## 3. Proceso del Concurso

### Fase Local - OnCampus (Enero - Febrero)

1. **Fase de Inscripción**:
   - Los equipos deberán registrarse antes del **31 de enero de 2025**.

2. **Fase de Desarrollo**:
   - Los equipos participarán en sesiones de mentoría y talleres durante febrero. La asistencia es obligatoria por al menos un miembro del equipo.
   - Tendrán la oportunidad de recibir retroalimentación directa para mejorar su modelo de negocio, impacto y pitch.
   - Tendrán acceso a **Hultie**, el **wsp agent** que brinda retroalimentación y resuelve dudas frecuentes.

3. **Evento Final OnCampus**:
   - **Fecha y formato**: Viernes 28 de febrero, presencial.
   - Cada equipo tendrá **4 minutos** para presentar su pitch ante un panel de jueces expertos, seguido de **4 minutos** de preguntas y respuestas.
   - **Certificados**: Los equipos que completen su pitch y de los que al menos un integrante haya asistido a las mentorías y talleres recibirán el certificado oficial de participación.
   - **Selección**: El equipo ganador será anunciado al final del evento final del OnCampus y representará a la universidad en la **National Competition**.

---

### National Competition (Mayo)
Los ganadores de cada OnCampus competirán en la **National Competition**. Los equipos presentarán un pitch en inglés ante un panel de expertos. Serán **4 minutos** de pitch y **4 minutos** de preguntas. El ganador nacional avanzará al **Hult Prize Digital Incubator**.

---

### Digital Incubator (Junio - Julio)
Hasta **60 startups globales** serán seleccionadas para participar en este programa intensivo, donde recibirán:

- **Mentorías personalizadas** con expertos globales.
- **Talleres** sobre diseño de impacto, validación de mercado, desarrollo de MVP y estrategias de escalamiento.
- **Recursos educativos digitales** y acceso a herramientas para desarrollar su empresa.

Al final del programa, las startups presentarán sus avances y los mejores equipos avanzarán al **Hult Prize Global Accelerator**.

---

### Global Accelerator (Agosto)
Hasta **25 startups** participarán en un programa presencial de un mes en **Ashridge House, Reino Unido**. Las startups recibirán:

- Acceso a inversionistas, socios potenciales y líderes de la industria.
- Múltiples **Demo Days** para recibir retroalimentación directa y preparar sus propuestas finales.

Al final del programa, las startups presentarán sus avances y las seis mejores startups serán seleccionadas para competir en la **Global Final**.

---

### Global Final (Septiembre)
Los seis equipos finalistas presentarán sus proyectos ante un panel de jueces internacionales el **5 de septiembre en Londres**. El equipo ganador recibirá el gran premio de **$1 millón USD** para escalar su startup.

---

## 4. Criterios de Evaluación

Los jueces evaluarán las propuestas con base en cuatro categorías principales:

1. **Equipo**:
   - **Organización**: Roles definidos y claridad en responsabilidades.
   - **Colaboración**: Capacidad de trabajar juntos para desarrollar y presentar la idea.
   - **Experiencia y habilidades**: Competencias complementarias y alineadas con el proyecto.

2. **Idea**:
   - **Identificación del problema**: Comprensión clara del problema social o ambiental que se busca resolver.
   - **Solución innovadora**: Idea creativa y viable.
   - **Validación**: Existencia de pruebas, retroalimentación de usuarios o prototipos iniciales.

3. **Impacto**:
   - **Alineación**: Relación directa con al menos un ODS.
   - **Medición**: Definición clara de KPIs y métricas de impacto.

4. **Viabilidad del Negocio**:
   - **Modelo de negocio**: Estructura clara, sostenible y realista.
   - **Ventaja competitiva**: Elementos diferenciadores frente a otras soluciones.

---

## 5. Mentorías y Recursos

- **Mentorías personalizadas**: Disponibles en febrero, con acceso a expertos locales e internacionales.
- **Talleres exclusivos**: Sesiones en vivo sobre creación de modelos de negocio, habilidades para pitching, etc.
- **Ponencias Nacionales**: Oportunidad de asistir a talleres de expertos reconocidos globalmente del ecosistema de innovación y startups.
- **Material grabado**: Acceso a grabaciones de talleres y recursos adicionales durante todo el programa.

---

## 6. Premios

1. **Primer Lugar en el OnCampus Program**:
   - Avance directo a la **Competencia Nacional Hult Prize**.
   - Certificado de excelencia.

2. **Otros premios**:
   - Certificados para todos los participantes que completen el programa.

---

## 7. Términos y Condiciones

1. La participación implica la aceptación de estas bases y del **Código de Conducta Hult Prize**.
2. Los participantes son responsables de los costos asociados con su participación (transporte, alimentación, etc.).
3. Al menos un miembro del equipo ganador debe asistir presencialmente al evento final.
4. Cualquier incumplimiento de las normas puede resultar en la descalificación inmediata.

---

**Contacto**:  
Si tienes alguna consulta o necesitas más información, no dudes en comunicarte con nosotros a través de los siguientes medios:  
- **Correo electrónico**: [hprizepucp@gmail.com](mailto:hprizepucp@gmail.com)  
- **Teléfono**: +51 949 147 463



"""
retriever_UCSM = """
# Bases del Concurso OnCampus Hult Prize  
### Universidad Católica de Santa María | Arequipa

## 1. Introducción
Hult Prize es la competencia estudiantil más grande del mundo que inspira a jóvenes emprendedores a resolver los mayores desafíos globales mediante startups sostenibles. Se conoce como el “Premio Nobel de los Estudiantes”. Desde su fundación, más de 1 millón de estudiantes han participado, buscando generar impacto positivo alineado con los **Objetivos de Desarrollo Sostenible (ODS)** de la ONU.

Para esta edición (2025), el desafío es **“Unlimited”**: Los equipos pueden elegir resolver cualquier problema, siempre y cuando la startup esté alineada con al menos un ODS.

**Campus Director**: Equipo administrativo

---

## 2. Elegibilidad y Registro

### Criterios de Elegibilidad:

1. **Estudiantes**:
   - Deben estar matriculados en algún curso/ciclo/especialización universitaria (pregrado o posgrado) y tener al menos 18 años cumplidos al 28 de febrero de 2025.

2. **Equipos**:
   - Cada equipo debe estar formado por **2 a 4 integrantes**.
   - Los equipos pueden incluir estudiantes de diferentes facultades o incluso universidades, pero deben representar a una sola universidad, con al menos un miembro inscrito en ella.
   - Todos los participantes deben estar registrados oficialmente en el formulario del equipo.

3. **Restricciones**:
   - Los equipos se pueden inscribir en solo un programa OnCampus: representar a solo una universidad.

### Proceso de Registro:

1. Completar el formulario de registro oficial antes del **31 de enero**.  
2. Confirmación de registro: recibirán un correo electrónico con detalles adicionales para la participación.

---

## 3. Proceso del Concurso

### Fase Local - OnCampus (Enero - Febrero)
El equipo administrativo detallará el proceso solo a los inscritos. Para cualquier consulta, escribir a [hultprizeperuoficial@gmail.com](mailto:hultprizeperuoficial@gmail.com) con el asunto **“Universidad Católica de Santa María | Arequipa”**.

### National Competition (Mayo)
Los ganadores de cada OnCampus competirán en la **National Competition**. Los equipos presentarán un pitch en inglés ante un panel de expertos. Serán **4 minutos** de pitch y **4 minutos** de preguntas. El ganador nacional avanzará al **Hult Prize Digital Incubator**.

### Digital Incubator (Junio - Julio)
Hasta **60 startups globales** serán seleccionadas para participar en este programa intensivo, donde recibirán:

- **Mentorías personalizadas** con expertos globales.
- **Talleres** sobre diseño de impacto, validación de mercado, desarrollo de MVP y estrategias de escalamiento.
- **Recursos educativos digitales** y acceso a herramientas para desarrollar su empresa.

Al final del programa, las startups presentarán sus avances y los mejores equipos avanzarán al **Hult Prize Global Accelerator**.

### Global Accelerator (Agosto)
Hasta **25 startups** participarán en un programa presencial de un mes en **Ashridge House, Reino Unido**. Las startups recibirán:

- Acceso a inversionistas, socios potenciales y líderes de la industria.
- Múltiples **Demo Days** para recibir retroalimentación directa y preparar sus propuestas finales.

Al final del programa, las startups presentarán sus avances y las seis mejores startups serán seleccionadas para competir en la **Global Final**.

### Global Final (Septiembre)
Los seis equipos finalistas presentarán sus proyectos ante un panel de jueces internacionales el **5 de septiembre en Londres**. El equipo ganador recibirá el gran premio de **$1 millón USD** para escalar su startup.

---

## 4. Criterios de Evaluación

Los jueces evaluarán las propuestas con base en cuatro categorías principales:

1. **Equipo**:
   - **Organización**: Roles definidos y claridad en responsabilidades.
   - **Colaboración**: Capacidad de trabajar juntos para desarrollar y presentar la idea.
   - **Experiencia y habilidades**: Competencias complementarias y alineadas con el proyecto.
   - **Escala**:
     - **1**: Desorganizado, roles poco claros y/o falta de experiencia.
     - **3**: Equipo promedio con roles básicos y comunicación moderada.
     - **5**: Equipo cohesionado, roles claros, excelente comunicación y complementariedad.

2. **Idea**:
   - **Identificación del problema**: Comprensión clara del problema social o ambiental que se busca resolver.
   - **Solución innovadora**: Idea creativa y viable.
   - **Validación**: Existencia de pruebas, retroalimentación de usuarios o prototipos iniciales.
   - **Escala**:
     - **1**: Problema poco definido, solución vaga y sin evidencia.
     - **3**: Problema claro, solución viable, con validación básica.
     - **5**: Solución probada, con retroalimentación robusta y potencial de impacto.

3. **Impacto**:
   - **Alineación**: Relación directa con al menos un ODS.
   - **Medición**: Definición clara de KPIs y métricas de impacto.
   - **Escalabilidad**: Potencial para expandir el impacto social a medida que crecen los ingresos.
   - **Escala**:
     - **1**: Sin alineación con ODS ni métricas claras.
     - **3**: Alineación con ODS y métricas iniciales.
     - **5**: Impacto claro y escalable, con KPIs bien definidos.

4. **Viabilidad del Negocio**:
   - **Modelo de negocio**: Estructura clara, sostenible y realista.
   - **Economía unitaria**: Comprensión de los costos, ingresos y rentabilidad.
   - **Ventaja competitiva**: Elementos diferenciadores frente a otras soluciones.
   - **Escala**:
     - **1**: Modelo débil y poco realista.
     - **3**: Modelo claro pero con áreas por fortalecer.
     - **5**: Modelo sólido, escalable y con alto potencial disruptivo.

---

## 5. Términos y Condiciones

1. La participación implica la aceptación de estas bases y del **Código de Conducta Hult Prize**.  
2. Los participantes son responsables de los costos asociados con su participación (transporte, alimentación, etc.).  
3. Al menos un miembro del equipo ganador debe asistir presencialmente al evento final.  
4. Cualquier incumplimiento de las normas puede resultar en la descalificación inmediata.

---

**Contacto**:  
[hultprizeperuoficial@gmail.com](mailto:hultprizeperuoficial@gmail.com)

"""


retriever_UTP = """
# Bases del Concurso OnCampus Hult Prize  
### Universidad Tecnológica del Perú  

---

## 1. Introducción

Hult Prize es la competencia estudiantil más grande del mundo que inspira a jóvenes emprendedores a resolver los mayores desafíos globales mediante startups sostenibles. Se conoce como el “Premio Nobel de los Estudiantes”. Desde su fundación más de 1 millón de estudiantes han participado, buscando generar impacto positivo alineado con los Objetivos de Desarrollo Sostenible (ODS) de la ONU.

Para esta edición (2025) el desafío es **“Unlimited”**: Los equipos pueden elegir resolver cualquier problema, siempre y cuando la startup esté alineada con al menos un ODS.

**Campus Director en UTP:** Giovanny Egoavil Cardenas

---

## 2. Elegibilidad y Registro

### Criterios de Elegibilidad:

**1. Estudiantes:**  
- Deben estar matriculados en algún curso/ciclo/especialización universitaria (pregrado o posgrado) y tener al menos 18 años cumplidos al 28 de febrero de 2025.

**2. Equipos:**  
- Cada equipo debe estar formado por **2 a 4 integrantes**.  
- Los equipos pueden incluir estudiantes de diferentes facultades o incluso universidades, pero deben representar a una sola universidad, con al menos un miembro inscrito en ella.  
- Todos los participantes deben estar registrados oficialmente en el formulario del equipo.

**3. Restricciones:**  
- Los equipos se pueden inscribir en solo un programa OnCampus: representará a solo una universidad.

### Proceso de Registro:

1. Completar el formulario de registro oficial antes del 31 de enero.  
2. Confirmación de registro: recibirán un correo electrónico con detalles adicionales para la participación.

---

## 3. Proceso del Concurso

### **Fase Local - OnCampus (Enero - Febrero)**

#### 1. Fase de Inscripción:  
Los equipos deben registrarse antes del 31 de enero de 2025.

#### 2. Fase de Desarrollo:  
- Los equipos participarán en sesiones de mentoría y talleres durante febrero. La asistencia es obligatoria por al menos un miembro del equipo.  
- Tendrán la oportunidad de recibir retroalimentación directa para mejorar su modelo de negocio, impacto y pitch.  
- Tendrán acceso a Hultie, el wsp agente que brinda retroalimentación y resuelve dudas.

#### 3. Evento Final OnCampus:  
- **Fecha y formato:** 22 de febrero (fecha tentativa), presencial.  
- Cada equipo tendrá 4 minutos para presentar su pitch ante un panel de jueces expertos, seguido de 4 minutos de preguntas y respuestas.  
- **Certificados:** Los equipos que completen su pitch y de los que al menos 1 integrante haya asistido a las mentorías y talleres recibirán el certificado oficial de participación.  
- **Selección:** El equipo ganador será anunciado al final del evento final del OnCampus y representará a la universidad en la **National Competition**.

---

### **National Competition (Mayo)**

Los ganadores de cada OnCampus competirán en la **National Competition**. Los equipos presentarán un pitch en inglés ante un panel de expertos. Serán 4 minutos de pitch y 4 de preguntas. El ganador nacional avanzará al **Hult Prize Digital Incubator**.

---

### **Digital Incubator (Junio - Julio)**

Hasta 60 startups globales serán seleccionadas para participar en este programa intensivo, donde recibirán:

- Mentorías personalizadas con expertos globales.  
- Talleres sobre diseño de impacto, validación de mercado, desarrollo de MVP y estrategias de escalamiento.  
- Recursos educativos digitales y acceso a herramientas para desarrollar su empresa.

Al final del programa, las startups presentarán sus avances y los mejores equipos avanzarán al **Hult Prize Global Accelerator**.

---

### **Global Accelerator (Agosto)**

Hasta 25 startups participarán en un programa presencial de un mes en Ashridge House, Reino Unido. Las startups recibirán:

- Acceso a inversionistas, socios potenciales y líderes de la industria.  
- Múltiples Demo Days para recibir retroalimentación directa y preparar sus propuestas finales.

Al final del programa, las startups presentarán sus avances y las seis mejores startups serán seleccionadas para competir en la **Global Final**.

---

### **Global Final (Septiembre)**

Los seis equipos finalistas presentarán sus proyectos ante un panel de jueces internacionales el 5 de septiembre en Londres. El equipo ganador recibirá el gran premio de **$1 millón USD** para escalar su startup.

---

## 4. Criterios de Evaluación

Los jueces evaluarán las propuestas con base en cuatro categorías principales:

### **1. Equipo:**  
- **Organización:** Roles definidos y claridad en responsabilidades.  
- **Colaboración:** Capacidad de trabajar juntos para desarrollar y presentar la idea.  
- **Experiencia y habilidades:** Competencias complementarias y alineadas con el proyecto.

**Escala:**  
- 0: Desorganizado, roles poco claros y/o falta de experiencia.  
- 3: Equipo promedio con roles básicos y comunicación moderada.  
- 5: Equipo cohesionado, roles claros, excelente comunicación y complementariedad.

---

### **2. Idea:**  
- **Identificación del problema:** Comprensión clara del problema social o ambiental que se busca resolver.  
- **Solución innovadora:** Idea creativa y viable.  
- **Validación:** Existencia de pruebas, retroalimentación de usuarios o prototipos iniciales.

**Escala:**  
- 0: Problema poco definido, solución vaga y sin evidencia.  
- 3: Problema claro, solución viable, con validación básica.  
- 5: Solución probada, con retroalimentación robusta y potencial de impacto.

---

### **3. Impacto:**  
- **Alineación:** Relación directa con al menos un ODS.  
- **Medición:** Definición clara de KPIs y métricas de impacto.  
- **Escalabilidad:** Potencial para expandir el impacto social a medida que crecen los ingresos.

**Escala:**  
- 0: Sin alineación con ODS ni métricas claras.  
- 3: Alineación con ODS y métricas iniciales.  
- 5: Impacto claro y escalable, con KPIs bien definidos.

---

### **4. Viabilidad del Negocio:**  
- **Modelo de negocio:** Estructura clara, sostenible y realista.  
- **Economía unitaria:** Comprensión de los costos, ingresos y rentabilidad.  
- **Ventaja competitiva:** Elementos diferenciadores frente a otras soluciones.

**Escala:**  
- 0: Modelo débil y poco realista.  
- 3: Modelo claro pero con áreas por fortalecer.  
- 5: Modelo sólido, escalable y con alto potencial disruptivo.

---

## 5. Mentorías y Recursos

- **Mentorías personalizadas:** Disponibles en febrero, con acceso a expertos locales e internacionales.  
- **Talleres exclusivos:** Sesiones en vivo sobre creación de modelos de negocio, habilidades para pitching, etc.  
- **Ponencias Nacionales:** Oportunidad de asistir a talleres de expertos reconocidos globalmente del ecosistema de innovación y startups.  
- **Material grabado:** Acceso a grabaciones de talleres y recursos adicionales durante todo el programa.

---

## 6. Términos y Condiciones

1. La participación implica la aceptación de estas bases y del **Código de Conducta Hult Prize**.  
2. Los participantes son responsables de los costos asociados con su participación (transporte, alimentación, etc.).  
3. Al menos un miembro del equipo ganador debe asistir presencialmente al evento final.  
4. Cualquier incumplimiento de las normas puede resultar en la descalificación inmediata.

---

## **Contacto**

- **Correo:** giovaego14@gmail.com  
- **Teléfono:** +51 929438735
"""




retrievers = {
    "CERTUS": retriever_CERTUS,
    "UCST": retriever_UCST,
    "UPC": retriever_UPC,
    "UPCH": retriever_UPCH,
    "UNCP": retriever_UNCP,
    "UCSM": retriever_UCSM,
    "UTP": retriever_UTP,
    "UNMSM": retriever_UNMSM,
    "UNAP": retriever_UNAP,
    "UNI": retriever_UNI,
    "UP": retriever_UP,
    "USIL": retriever_USIL,
    "UDEP": retriever_UDEP,
    "UCSUR": retriever_UCSUR,
    "UARM": retriever_UARM,
    "ULIMA": retriever_ULIMA,
    "ESAN": retriever_ESAN,
    "UCAL": retriever_UCAL,
    "UTEC": retriever_UTEC,
    "PUCP": retriever_PUCP
}




def get_retriever(sede):
    try:
        retriever = retrievers.get(sede)
        if retriever:
            return retriever
        else:
            return "Todavia no se publicaron las bases de esta sede"
    except:
        return "Todavia no se publicaron las bases de esta sede"
