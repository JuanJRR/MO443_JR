import argparse
import os
import sys

import cv2
from trabalho4_geometric import rotate_image, scale_image
from trabalho4_panorama import process_panorama


class WorkflowManager:
    """Gestor del flujo de trabajo completo."""

    @staticmethod
    def validate_image_path(path):
        """Validar que la ruta de imagen existe."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Imagen no encontrada: {path}")
        return path

    @staticmethod
    def validate_output_dir(path):
        """Validar y crear directorio de salida si no existe."""
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def execute_transformation(args):
        """Ejecutar transformación geométrica."""
        print(f"\n{'=' * 60}")
        print("TRANSFORMACIONES GEOMÉTRICAS")
        print(f"{'=' * 60}")

        input_img = WorkflowManager.validate_image_path(args.input)
        output_dir = WorkflowManager.validate_output_dir(args.output_dir)

        img = cv2.imread(input_img)
        if img is None:
            raise ValueError(f"No se puede cargar: {input_img}")

        print(f"Imagen de entrada: {input_img}")
        print(f"Tamaño: {img.shape[1]}x{img.shape[0]}")

        if args.mode == "scale":
            print(f"Modo: Escalado ({args.scale}x)")
            print(f"Método de interpolación: {args.method}")

            result = scale_image(img, args.scale, args.method)

        elif args.mode == "rotate":
            print(f"Modo: Rotación ({args.angle}°)")
            print(f"Método de interpolación: {args.method}")

            result = rotate_image(img, args.angle, args.method)

        output_path = os.path.join(output_dir, "transformacion_resultado.png")
        cv2.imwrite(output_path, result)
        print(f"Imagen de salida: {output_path}")
        print(f"Tamaño: {result.shape[1]}x{result.shape[0]}")

        return result

    @staticmethod
    def execute_panorama(args):
        """Ejecutar construcción de panorama."""
        print(f"\n{'=' * 60}")
        print("CONSTRUCCIÓN DE IMAGEN PANORÁMICA")
        print(f"{'=' * 60}")

        img1_path = WorkflowManager.validate_image_path(args.input1)
        img2_path = WorkflowManager.validate_image_path(args.input2)
        output_dir = WorkflowManager.validate_output_dir(args.output_dir)

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        print(f"Imagen 1: {img1_path} ({img1.shape[1]}x{img1.shape[0]})")
        print(f"Imagen 2: {img2_path} ({img2.shape[1]}x{img2.shape[0]})")
        print(f"Método de detección: {args.feature}")
        print(f"Umbral de ratio: {args.ratio_threshold}")
        print(f"Umbral RANSAC: {args.ransac_threshold}")

        matches_output = os.path.join(output_dir, "correspondencias.jpg")
        panorama_output = os.path.join(output_dir, "panorama.jpg")

        result = process_panorama(
            img1_path,
            img2_path,
            matches_output,
            panorama_output,
            feature_method=args.feature,
            matcher_method="brute_force",
            ratio_threshold=args.ratio_threshold,
            ransac_threshold=args.ransac_threshold,
        )

        print(f"\nResultados:")
        print(f"  Correspondencias encontradas: {result['num_matches']}")
        print(f"  Salida (correspondencias): {matches_output}")
        print(f"  Salida (panorama): {panorama_output}")
        print(
            f"  Tamaño panorama: {result['panorama'].shape[1]}x{result['panorama'].shape[0]}"
        )
        print(f"\nMatriz de homografía:")
        print(result["H"])

        return result

    @staticmethod
    def execute_full_pipeline(args):
        """Ejecutar pipeline completa."""
        print(f"\n{'=' * 60}")
        print("PIPELINE COMPLETA")
        print(f"{'=' * 60}")

        output_dir = WorkflowManager.validate_output_dir(args.output_dir)

        # Parte 1: Transformaciones geométricas
        print("\n[1/2] Ejecutando transformaciones geométricas...")

        args.input = args.input_transform
        args.output_dir = output_dir

        try:
            transform_result = WorkflowManager.execute_transformation(args)
            print("✓ Transformaciones completadas")
        except Exception as e:
            print(f"✗ Error en transformaciones: {e}")
            return

        # Parte 2: Panorama
        if args.input1 and args.input2:
            print("\n[2/2] Ejecutando construcción de panorama...")

            try:
                panorama_result = WorkflowManager.execute_panorama(args)
                print("✓ Panorama completado")
            except Exception as e:
                print(f"✗ Error en panorama: {e}")
                return

        print(f"\n{'=' * 60}")
        print("PIPELINE COMPLETA FINALIZADA EXITOSAMENTE")
        print(f"{'=' * 60}")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Trabalho 4: Procesamiento Digital de Imágenes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

Transformación geométrica (escalado):
  python main.py transform -i imagen.png -e 2.5 -m bilinear -o ./output

Transformación geométrica (rotación):
  python main.py transform -i imagen.png -a 45 -m bicubic -o ./output

Construcción de panorama:
  python main.py panorama -i1 imagen1.jpg -i2 imagen2.jpg -f sift -o ./output

Pipeline completa:
  python main.py pipeline -i imagen.png -e 1.5 -i1 img1.jpg -i2 img2.jpg -o ./output
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Comando a ejecutar")

    # Subcomando: transform
    transform_parser = subparsers.add_parser(
        "transform", help="Transformaciones geométricas"
    )
    transform_parser.add_argument(
        "-i", "--input", required=True, help="Imagen de entrada (PNG)"
    )
    transform_parser.add_argument(
        "-o", "--output-dir", default="./output", help="Directorio de salida"
    )
    transform_parser.add_argument(
        "-a", "--angle", type=float, default=0, help="Ángulo de rotación en grados"
    )
    transform_parser.add_argument(
        "-e", "--scale", type=float, default=1, help="Factor de escala"
    )
    transform_parser.add_argument(
        "-m",
        "--method",
        default="bilinear",
        choices=["nearest", "bilinear", "bicubic", "lagrange"],
        help="Método de interpolación",
    )
    transform_parser.add_argument(
        "--mode",
        choices=["scale", "rotate"],
        help="Modo de operación (auto-detectado si no se especifica)",
    )

    # Subcomando: panorama
    panorama_parser = subparsers.add_parser(
        "panorama", help="Construcción de imagen panorámica"
    )
    panorama_parser.add_argument(
        "-i1", "--input1", required=True, help="Primera imagen (JPEG)"
    )
    panorama_parser.add_argument(
        "-i2", "--input2", required=True, help="Segunda imagen (JPEG)"
    )
    panorama_parser.add_argument(
        "-o", "--output-dir", default="./output", help="Directorio de salida"
    )
    panorama_parser.add_argument(
        "-f",
        "--feature",
        default="sift",
        choices=["sift", "surf", "orb", "brief"],
        help="Método de detección de características",
    )
    panorama_parser.add_argument(
        "-t",
        "--ratio-threshold",
        type=float,
        default=0.7,
        help="Umbral de ratio para correspondencias",
    )
    panorama_parser.add_argument(
        "-r", "--ransac-threshold", type=float, default=5.0, help="Umbral de RANSAC"
    )

    # Subcomando: pipeline
    pipeline_parser = subparsers.add_parser("pipeline", help="Pipeline completa")
    pipeline_parser.add_argument(
        "-i", "--input-transform", help="Imagen para transformación"
    )
    pipeline_parser.add_argument(
        "-a", "--angle", type=float, default=0, help="Ángulo de rotación"
    )
    pipeline_parser.add_argument(
        "-e", "--scale", type=float, default=1, help="Factor de escala"
    )
    pipeline_parser.add_argument(
        "-m",
        "--method",
        default="bilinear",
        choices=["nearest", "bilinear", "bicubic", "lagrange"],
    )
    pipeline_parser.add_argument("-i1", "--input1", help="Primera imagen para panorama")
    pipeline_parser.add_argument("-i2", "--input2", help="Segunda imagen para panorama")
    pipeline_parser.add_argument(
        "-o", "--output-dir", default="./output", help="Directorio de salida"
    )
    pipeline_parser.add_argument(
        "-f", "--feature", default="sift", choices=["sift", "surf", "orb", "brief"]
    )
    pipeline_parser.add_argument("-t", "--ratio-threshold", type=float, default=0.7)
    pipeline_parser.add_argument("-r", "--ransac-threshold", type=float, default=5.0)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "transform":
            if args.angle != 0 and args.scale != 1:
                print("Error: especifique solo uno de -a (ángulo) o -e (escala)")
                sys.exit(1)

            if not args.mode:
                args.mode = "rotate" if args.angle != 0 else "scale"

            WorkflowManager.execute_transformation(args)

        elif args.command == "panorama":
            WorkflowManager.execute_panorama(args)

        elif args.command == "pipeline":
            WorkflowManager.execute_full_pipeline(args)

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
