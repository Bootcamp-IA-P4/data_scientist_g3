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
    """Ejecutar script SQL usando conexión directa a PostgreSQL"""
    try:
        cursor = connection.cursor()
        
        # Limpiar y dividir comandos de forma más robusta
        # Eliminar comentarios línea por línea primero
        lines = []
        for line in sql_script.split('\n'):
            line = line.strip()
            if line and not line.startswith('--'):
                lines.append(line)
        
        # Unir líneas y dividir por punto y coma
        clean_script = ' '.join(lines)
        commands = [cmd.strip() for cmd in clean_script.split(';') if cmd.strip()]
        
        print(f"📋 Ejecutando {len(commands)} comandos SQL...")
        
        # Mostrar los primeros caracteres de cada comando para debug
        for i, command in enumerate(commands, 1):
            command_preview = command[:80].replace('\n', ' ')
            print(f"   [{i}/{len(commands)}] {command_preview}...")
        
        print("\n🔧 Ejecutando comandos...")
        
        for i, command in enumerate(commands, 1):
            print(f"   [{i}/{len(commands)}] Ejecutando...")
            
            # DEBUG: Imprimir comando completo si falla
            try:
                cursor.execute(command)
                connection.commit()
                print(f"   ✅ Comando {i} ejecutado correctamente")
            except psycopg2.Error as e:
                print(f"   ❌ ERROR EN COMANDO {i}:")
                print(f"   COMANDO COMPLETO: {repr(command)}")
                print(f"   ERROR: {e}")
                
                # Manejar errores comunes
                error_msg = str(e).lower()
                if "already exists" in error_msg:
                    print(f"   ⚠️ Objeto ya existe (OK)")
                    connection.rollback()  # Rollback para continuar
                elif "does not exist" in error_msg and ("comment on" in command.lower()):
                    print(f"   ⚠️ Comentario omitido")
                    connection.rollback()
                else:
                    connection.rollback()
        
        cursor.close()
        return True
        
    except Exception as e:
        print(f"❌ Error ejecutando SQL: {e}")
        connection.rollback()
        return False

def verify_table_creation(connection) -> bool:
    """Verificar que la tabla se creó correctamente"""
    try:
        cursor = connection.cursor()
        
        # Verificar que la tabla existe
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'stroke_predictions'
            );
        """)
        
        table_exists = cursor.fetchone()[0]
        
        if table_exists:
            # Contar registros existentes
            cursor.execute("SELECT COUNT(*) FROM stroke_predictions;")
            count = cursor.fetchone()[0]
            print(f"✅ Tabla 'stroke_predictions' creada correctamente ({count} registros)")
            
            # Verificar estructura básica
            cursor.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'stroke_predictions' 
                ORDER BY ordinal_position;
            """)
            columns = cursor.fetchall()
            print(f"📊 Tabla tiene {len(columns)} columnas")
            
            cursor.close()
            return True
        else:
            print("❌ La tabla no fue creada")
            cursor.close()
            return False
        
    except Exception as e:
        print(f"❌ Error verificando tabla: {e}")
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
        print("\n🔍 Verificando creación de tablas...")
        if verify_table_creation(connection):
            print("✅ Verificación exitosa")
        else:
            print("❌ Error en verificación")
            sys.exit(1)
        
        print("\n" + "=" * 50)
        print("🎉 ¡Base de datos configurada exitosamente!")
        print("📊 Tabla 'stroke_predictions' lista para usar")
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