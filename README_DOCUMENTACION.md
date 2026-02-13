# Documentación del Sistema Esclavo - Descifrado de Imágenes con Caos

## Descripción

Este documento técnico (`Documentacion_Sistema_Esclavo.tex`) proporciona una explicación detallada y paso a paso del proceso de descifrado de imágenes implementado en el archivo `Esclavo_TLS KEYSTREAM.py`.

## Contenido del Documento

La documentación cubre las siguientes fases del proceso de descifrado:

### 1. **Conexión Segura mediante MQTT con TLS**
   - Configuración del protocolo MQTT sobre TLS
   - Autenticación y cifrado de comunicaciones
   - Suscripción a topics para recepción de datos

### 2. **Recepción y Extracción de Parámetros**
   - Parámetros del oscilador de Rössler
   - Parámetros del mapa logístico
   - Datos cifrados y trayectorias del sistema maestro

### 3. **Sincronización del Oscilador de Rössler**
   - Ecuaciones diferenciales del oscilador esclavo
   - Proceso de integración numérica
   - Extracción del keystream caótico sincronizado

### 4. **Generación del Mapa Logístico**
   - Ecuación iterativa del mapa logístico
   - Regeneración del vector de permutación

### 5. **Reversión de la Confusión**
   - Proceso matemático de sustracción de componentes
   - Recuperación del vector de difusión
   - Escalado inverso

### 6. **Reversión de la Difusión**
   - Algoritmo de permutación inversa
   - Manejo de colisiones
   - Recuperación del orden original de píxeles

### 7. **Reconstrucción de la Imagen**
   - Conversión de datos normalizados a píxeles
   - Reestructuración dimensional (250×250×3)
   - Generación de imagen RGB

### 8. **Validación y Métricas de Calidad**
   - Errores de sincronización
   - Distancia de Hamming
   - Coeficiente de correlación

### 9. **Análisis de Sensibilidad (Opcional)**
   - Sensibilidad a parámetros de Rössler
   - Sensibilidad a parámetros del mapa logístico

## Características del Documento

- **Idioma:** Español
- **Formato:** LaTeX (compilable a PDF)
- **Tablas de ejemplo:** Incluye 10 valores de muestra para cada etapa del proceso
- **Contexto:** Imagen RGB de 250×250 píxeles (187,500 elementos)
- **Ecuaciones matemáticas:** Totalmente documentadas con notación LaTeX
- **Técnicas criptográficas:** Explicación detallada de confusión y difusión

## Tablas Incluidas

El documento incluye múltiples tablas con ejemplos de datos en cada fase:

1. **Configuración MQTT con TLS**
2. **Parámetros del oscilador de Rössler**
3. **Parámetros del mapa logístico**
4. **Vector cifrado recibido** (primeros 10 valores)
5. **Trayectoria sincronizada** del oscilador (primeros 10 valores)
6. **Vector logístico regenerado** (primeros 10 valores)
7. **Vector de mezcla** (primeros 10 valores)
8. **Proceso de reversión de confusión** (primeros 10 valores)
9. **Proceso de reversión de difusión** (primeros 10 valores)
10. **Vector reconstruido final** (primeros 10 valores)

Todos los ejemplos utilizan datos coherentes con el sistema maestro proporcionado.

## Compilación del Documento

### Requisitos

Para compilar el documento LaTeX necesitas:

```bash
# En Ubuntu/Debian
sudo apt-get install texlive-latex-base texlive-latex-extra texlive-lang-spanish

# En Fedora/RHEL
sudo dnf install texlive-scheme-basic texlive-collection-langspanish

# En Arch Linux
sudo pacman -S texlive-core texlive-latexextra
```

### Compilación

```bash
# Compilar el documento (ejecutar 2 veces para referencias correctas)
pdflatex Documentacion_Sistema_Esclavo.tex
pdflatex Documentacion_Sistema_Esclavo.tex
```

El archivo `Documentacion_Sistema_Esclavo.pdf` será generado en el mismo directorio.

## Archivos Generados

- `Documentacion_Sistema_Esclavo.tex` - Código fuente LaTeX
- `Documentacion_Sistema_Esclavo.pdf` - Documento compilado (17 páginas)
- `.gitignore` - Excluye archivos auxiliares de LaTeX del control de versiones

## Uso del Documento

Este documento está diseñado para:

- **Documentación técnica:** Referencia completa del proceso de descifrado
- **Educación:** Material didáctico sobre criptografía caótica
- **Análisis:** Base para entender el funcionamiento del sistema
- **Publicaciones:** Puede ser incluido en artículos académicos o reportes técnicos

## Estructura del Sistema

```
Sistema Maestro (Cifrado)
    ↓
[MQTT + TLS] - Comunicación Segura
    ↓
Sistema Esclavo (Descifrado)
    ├── Fase 1: Conexión MQTT/TLS
    ├── Fase 2: Recepción de datos
    ├── Fase 3: Sincronización Rössler
    ├── Fase 4: Generación mapa logístico
    ├── Fase 5: Reversión confusión
    ├── Fase 6: Reversión difusión
    ├── Fase 7: Reconstrucción imagen
    └── Fase 8: Validación
```

## Técnicas Criptográficas Documentadas

1. **Comunicación segura** - TLS con autenticación por certificados
2. **Sincronización caótica** - Regeneración del keystream sin transmisión directa
3. **Generador pseudoaleatorio** - Mapa logístico determinista
4. **Confusión** - Keystream cipher con sistemas caóticos
5. **Difusión** - Permutación inversa de píxeles
6. **Sensibilidad paramétrica** - Seguridad basada en caos

## Contacto y Contribuciones

Este documento fue generado automáticamente como parte del sistema de cifrado maestro-esclavo para la Raspberry Pi 4.

## Licencia

El documento se proporciona como parte del proyecto CifradoSlave.
