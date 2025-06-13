"""
Script para crear automáticamente las tablas en PostgreSQL
Conexión directa a la base de datos usando psycopg2
"""

import os
import sys
from pathlib import Path
import psycopg2
from dotenv import load_dotenv

# Cargar variables de entorno desde la raíz del proyecto (forzar sobreescritura)
load_dotenv(Path(__file__).parent.parent / '.env', override=True)

def get_db_connection():
    """Crear conexión directa a PostgreSQL"""
    database_url = os.getenv("DATABASE_URL")
    
    if not database_url:
        raise ValueError(
            "Falta DATABASE_URL en .env:\n"
            "- DATABASE_URL=postgresql://postgres:password@host:port/database"
        )
    
    print(f"🔗 Conectando a PostgreSQL...")
    return psycopg2.connect(database_url)

def read_schema_file() -> str:
    """Leer el archivo schema.sql"""
    schema_path = Path(__file__).parent / "schema.sql"
    
    if not schema_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {schema_path}")
    
    with open(schema_path, 'r', encoding='utf-8') as file:
        return file.read()

def execute_sql_script(connection, sql_script: str) -> bool:
    """Ejecutar script SQL con mejor parsing para funciones PostgreSQL"""
    try:
        cursor = connection.cursor()
        
        # Ejecutar todo el script de una vez para mantener el contexto
        print("🔧 Ejecutando script SQL completo...")
        print("📋 Creando secuencia, tabla, índices, función y vista...")
        
        try:
            cursor.execute(sql_script)
            connection.commit()
            print("✅ Script ejecutado correctamente")
            
            # Verificar componentes creados
            cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'stroke_predictions';")
            table_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM information_schema.views WHERE table_name = 'stroke_predictions_formatted';")
            view_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM information_schema.routines WHERE routine_name = 'format_date_spanish';")
            function_count = cursor.fetchone()[0]
            
            print(f"📊 Componentes creados:")
            print(f"   - Tabla 'stroke_predictions': {'✅' if table_count > 0 else '❌'}")
            print(f"   - Vista 'stroke_predictions_formatted': {'✅' if view_count > 0 else '❌'}")
            print(f"   - Función 'format_date_spanish': {'✅' if function_count > 0 else '❌'}")
            
            cursor.close()
            return True
            
        except psycopg2.Error as e:
            print(f"❌ Error ejecutando script: {e}")
            connection.rollback()
            
            # Si falla todo junto, intentar por partes
            print("🔄 Intentando ejecutar por componentes...")
            return execute_sql_by_components(connection, sql_script)
        
    except Exception as e:
        print(f"❌ Error general ejecutando SQL: {e}")
        connection.rollback()
        return False

def execute_sql_by_components(connection, sql_script: str) -> bool:
    """Ejecutar SQL dividido en componentes lógicos"""
    try:
        cursor = connection.cursor()
        
        # Dividir el script en componentes lógicos
        components = [
            "CREATE SEQUENCE",
            "CREATE TABLE", 
            "CREATE INDEX",
            "CREATE OR REPLACE FUNCTION",
            "CREATE VIEW",
            "COMMENT ON"
        ]
        
        lines = sql_script.split('\n')
        current_component = []
        component_type = ""
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('--'):
                continue
                
            # Detectar inicio de nuevo componente
            for comp in components:
                if line.startswith(comp):
                    # Ejecutar componente anterior si existe
                    if current_component and component_type:
                        execute_component(cursor, connection, current_component, component_type)
                    
                    # Iniciar nuevo componente
                    current_component = [line]
                    component_type = comp
                    break
            else:
                # Continuar con el componente actual
                if current_component:
                    current_component.append(line)
        
        # Ejecutar último componente
        if current_component and component_type:
            execute_component(cursor, connection, current_component, component_type)
        
        cursor.close()
        return True
        
    except Exception as e:
        print(f"❌ Error ejecutando por componentes: {e}")
        return False

def execute_component(cursor, connection, component_lines, component_type):
    """Ejecutar un componente individual del SQL"""
    try:
        component_sql = ' '.join(component_lines)
        
        # Para funciones, manejar correctamente los delimitadores $$
        if "FUNCTION" in component_type:
            # Asegurarse de que la función termine correctamente
            if not component_sql.rstrip().endswith(';'):
                component_sql += ';'
        
        print(f"   Ejecutando {component_type}...")
        cursor.execute(component_sql)
        connection.commit()
        print(f"   ✅ {component_type} creado correctamente")
        
    except psycopg2.Error as e:
        error_msg = str(e).lower()
        if "already exists" in error_msg:
            print(f"   ⚠️ {component_type} ya existe (OK)")
            connection.rollback()
        else:
            print(f"   ❌ Error en {component_type}: {e}")
            connection.rollback()

def verify_table_creation(connection) -> bool:
    """Verificar que todos los componentes se crearon correctamente"""
    try:
        cursor = connection.cursor()
        
        # Verificar tabla stroke_predictions
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'stroke_predictions'
            );
        """)
        stroke_table_exists = cursor.fetchone()[0]

        # Verificar tabla image_predictions
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'image_predictions'
            );
        """)
        image_table_exists = cursor.fetchone()[0]
        
        # Verificar vistas
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.views 
                WHERE table_schema = 'public' 
                AND table_name = 'stroke_predictions_formatted'
            );
        """)
        stroke_view_exists = cursor.fetchone()[0]

        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.views 
                WHERE table_schema = 'public' 
                AND table_name = 'image_predictions_formatted'
            );
        """)
        image_view_exists = cursor.fetchone()[0]
        
        # Verificar función
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.routines 
                WHERE routine_schema = 'public' 
                AND routine_name = 'format_date_spanish'
            );
        """)
        function_exists = cursor.fetchone()[0]
        
        if all([stroke_table_exists, image_table_exists, stroke_view_exists, image_view_exists, function_exists]):
            # Contar registros existentes
            cursor.execute("SELECT COUNT(*) FROM stroke_predictions;")
            stroke_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM image_predictions;")
            image_count = cursor.fetchone()[0]
            
            # Probar la vista formateada
            cursor.execute("SELECT format_date_spanish(NOW());")
            formatted_date = cursor.fetchone()[0]
            
            print(f"✅ Tabla 'stroke_predictions' creada correctamente ({stroke_count} registros)")
            print(f"✅ Tabla 'image_predictions' creada correctamente ({image_count} registros)")
            print(f"✅ Vista 'stroke_predictions_formatted' creada correctamente")
            print(f"✅ Vista 'image_predictions_formatted' creada correctamente")
            print(f"✅ Función 'format_date_spanish' creada correctamente")
            print(f"📅 Fecha de ejemplo: {formatted_date}")
            
            cursor.close()
            return True
        else:
            missing = []
            if not stroke_table_exists: missing.append("tabla stroke_predictions")
            if not image_table_exists: missing.append("tabla image_predictions")
            if not stroke_view_exists: missing.append("vista stroke_predictions_formatted")
            if not image_view_exists: missing.append("vista image_predictions_formatted")
            if not function_exists: missing.append("función format_date_spanish")
            
            print(f"❌ Faltan componentes: {', '.join(missing)}")
            cursor.close()
            return False
        
    except Exception as e:
        print(f"❌ Error verificando componentes: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 Iniciando setup de base de datos PostgreSQL...")
    print("-" * 50)
    
    connection = None
    
    try:
        # 1. Conectar a PostgreSQL
        print("📡 Conectando a PostgreSQL...")
        connection = get_db_connection()
        print("✅ Conexión establecida")
        
        # 2. Leer schema
        print("\n📄 Leyendo archivo schema.sql...")
        sql_script = read_schema_file()
        print("✅ Schema cargado")
        
        # 3. Ejecutar SQL
        print("\n🔧 Ejecutando comandos SQL...")
        if execute_sql_script(connection, sql_script):
            print("✅ Schema ejecutado correctamente")
        else:
            print("❌ Error ejecutando schema")
            sys.exit(1)
        
        # 4. Verificar creación
        print("\n🔍 Verificando creación de componentes...")
        if verify_table_creation(connection):
            print("✅ Verificación exitosa")
        else:
            print("❌ Error en verificación")
            sys.exit(1)
        
        print("\n" + "=" * 50)
        print("🎉 ¡Base de datos configurada exitosamente!")
        print("📊 Tabla 'stroke_predictions' lista para usar")
        print("📸 Tabla 'image_predictions' lista para predicciones de imágenes")
        print("📅 Vista formateadas con fechas españolas")
        print("🔗 Conexión directa a PostgreSQL funcionando")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Error general: {e}")
        sys.exit(1)
    
    finally:
        # Cerrar conexión
        if connection:
            connection.close()
            print("🔌 Conexión cerrada")

if __name__ == "__main__":
    main()