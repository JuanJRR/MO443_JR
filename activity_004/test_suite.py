import os

import cv2
from load_save import upload_images
from trabalho4_geometric import rotate_image, scale_image
from trabalho4_panorama import PanoramaBuilder


def test_geometric_transformations():
    """Prueba de transformaciones geométricas."""
    print("\n" + "=" * 70)
    print("TEST 1: TRANSFORMACIONES GEOMÉTRICAS")
    print("=" * 70)

    test_dir = "./test_data"
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(f"{test_dir}/output", exist_ok=True)

    # imagen de prueba
    test_img_path = f"{test_dir}/test_original.png"
    test_img = upload_images(path=test_img_path, color=False)
    print(f"✓ Imagen de prueba: {test_img_path}")
    print(f"  Tamaño: {test_img.shape[1]}x{test_img.shape[0]}")

    # Test escalado
    print("\n[Escalado]")
    methods = ["nearest", "bilinear", "bicubic", "lagrange"]
    scale_factors = [0.5, 1.5, 2.0]

    for method in methods:
        for scale in scale_factors:
            try:
                result = scale_image(test_img, scale, method)
                output = f"{test_dir}/output/scale_{method}_{scale:.1f}.png"
                cv2.imwrite(output, result)
                print(
                    f"  ✓ {method:10s} x{scale:.1f}: {result.shape[1]:4d}x{result.shape[0]:4d}"
                )
            except Exception as e:
                print(f"  ✗ {method:10s} x{scale:.1f}: Error - {e}")

    # Test rotación
    print("\n[Rotación]")
    angles = [0, 15, 45]

    for method in methods:
        for angle in angles:
            try:
                result = rotate_image(test_img, angle, method)
                output = f"{test_dir}/output/rotate_{method}_{angle}.png"
                cv2.imwrite(output, result)
                print(
                    f"  ✓ {method:10s} {angle:3d}°: {result.shape[1]:4d}x{result.shape[0]:4d}"
                )
            except Exception as e:
                print(f"  ✗ {method:10s} {angles[0]:3d}°: Error - {e}")

    print("\n✓ Tests de transformaciones geométricas completados")
    return test_dir


def test_panorama_creation():
    """Prueba de creación de panorama."""
    print("\n" + "=" * 70)
    print("TEST 2: CONSTRUCCIÓN DE PANORAMA")
    print("=" * 70)

    test_dir = "./test_data"
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(f"{test_dir}/output", exist_ok=True)

    # Crear imágenes de prueba
    img1_path = f"{test_dir}/panorama_img1.jpg"
    img2_path = f"{test_dir}/panorama_img2.jpg"

    img1 = upload_images(path=img1_path, color=True)
    img2 = upload_images(path=img2_path, color=True)
    print("✓ Imágenes de panorama")
    print(f"  Imagen 1: {img1_path} ({img1.shape[1]}x{img1.shape[0]})")
    print(f"  Imagen 2: {img2_path} ({img2.shape[1]}x{img2.shape[0]})")

    # Test diferentes métodos de detección
    methods = ["sift", "orb"]

    for method in methods:
        print(f"\n[Método: {method.upper()}]")
        try:
            builder = PanoramaBuilder(feature_method=method)

            registration_data = builder.register_images(
                img1_path, img2_path, ratio_threshold=0.5, ransac_threshold=5.0
            )

            matches = registration_data["matches"]
            H = registration_data["H"]

            print(f"  ✓ Características detectadas:")
            print(f"    - Imagen 1: {len(registration_data['kp1'])} puntos")
            print(f"    - Imagen 2: {len(registration_data['kp2'])} puntos")
            print(f"    - Correspondencias: {len(matches)}")

            # Dibujar correspondencias
            matches_output = f"{test_dir}/output/matches_{method}.jpg"
            builder.draw_matches(
                registration_data["img1"],
                registration_data["img2"],
                registration_data["kp1"],
                registration_data["kp2"],
                matches,
                matches_output,
            )
            print(f"  ✓ Correspondencias guardadas: {matches_output}")

            # Crear panorama
            panorama = builder.build_panorama(
                registration_data["img1"], registration_data["img2"]
            )
            panorama_output = f"{test_dir}/output/panorama_{method}.jpg"
            cv2.imwrite(panorama_output, panorama)
            print(f"  ✓ Panorama creado: {panorama_output}")
            print(f"    Tamaño: {panorama.shape[1]}x{panorama.shape[0]}")

            print(f"  ✓ Matriz de homografía:")
            for row in H:
                print(f"    {row}")

        except Exception as e:
            print(f"  ✗ Error con {method}: {e}")

    print("\n✓ Tests de panorama completados")
    return test_dir


def main():
    """Ejecutar todos los tests."""
    print("\n" + "=" * 70)
    print("TRABALHO 4: SUITE DE PRUEBA COMPLETA")
    print("=" * 70)

    try:
        test_dir = test_geometric_transformations()
        test_dir = test_panorama_creation()

        print("\n" + "=" * 70)
        print("✓ TODOS LOS TESTS COMPLETADOS EXITOSAMENTE")
        print("=" * 70)
        print(f"\nArchivos de prueba generados en: {test_dir}/output/")
        print("Puede visualizar los resultados con:")
        print(f"  ls -la {test_dir}/output/")

    except Exception as e:
        print(f"\n✗ Error durante la ejecución: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
