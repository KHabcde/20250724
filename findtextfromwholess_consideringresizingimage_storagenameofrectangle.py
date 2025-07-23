import os
import base64
from dotenv import load_dotenv
from cmd_rectangle_coordinate_strage import cmd_rectangle_coordinate_strage

# 会社用 (Azure)
from openai import AzureOpenAI

# 家用 (ChatGPT)
from openai import OpenAI

# Load environment variables
load_dotenv()

def findtextfromwholess_consideringresizingimage_storagenameofrectangle(prompt, image_path, storage_name):
    """
    AI(LLM)を使って画像から指定されたテキストやUI要素の矩形領域を特定し、
    座標を補正・拡張してから保存する機能モジュール
    
    Args:
        prompt (str): AIに送るプロンプト（探したい要素の説明）
        image_path (str): 画像のファイルパス
        storage_name (str): 矩形領域を保存する名前
    """
    try:
        # 第一ステップ: LLMへプロンプトと画像を渡す
        rectangle_coords = call_llm_for_rectangle_detection(prompt, image_path)
        
        if not rectangle_coords:
            print(f"警告: LLMから矩形領域を取得できませんでした。デフォルト値で続行します。")
            rectangle_coords = {
                'x1': 100.0,
                'y1': 100.0,
                'x2': 200.0,
                'y2': 200.0
            }
        
        print(f"取得した座標: {rectangle_coords}")
        
        # 第二ステップ: LLMが出力した矩形領域の値の変換 (1.41倍)
        scaled_coords = scale_coordinates(rectangle_coords, 1.41)
        print(f"1.41倍後の座標: {scaled_coords}")
        
        # 第三ステップ: 縦横の長さを2倍に拡張（バッファー）
        buffered_coords = expand_rectangle(scaled_coords, 2.0)
        print(f"バッファー適用後の座標: {buffered_coords}")
        
        # 第四ステップ: 座標系の登録
        cmd_rectangle_coordinate_strage(
            storage_name,
            str(int(buffered_coords['x1'])),
            str(int(buffered_coords['y1'])),
            str(int(buffered_coords['x2'])),
            str(int(buffered_coords['y2']))
        )
        
        print(f"矩形領域を '{storage_name}' として保存しました")
        
    except Exception as e:
        print(f"Error in findtextfromwholess_consideringresizingimage_storagenameofrectangle: {e}")
        # エラーが発生した場合でもデフォルト座標で処理を続行
        try:
            print("エラー時のデフォルト処理を実行します")
            cmd_rectangle_coordinate_strage(storage_name, "100", "100", "300", "300")
            print(f"デフォルト矩形領域を '{storage_name}' として保存しました")
        except Exception as fallback_error:
            print(f"フォールバック処理も失敗しました: {fallback_error}")

def call_llm_for_rectangle_detection(prompt, image_path):
    """
    LLMを呼び出して画像から矩形領域を検出する
    
    Args:
        prompt (str): AIに送るプロンプト
        image_path (str): 画像のファイルパス
        
    Returns:
        dict: 矩形領域の座標 {'x1': float, 'y1': float, 'x2': float, 'y2': float}
    """
    try:
        # 画像をbase64エンコード
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # 会社用 (Azure) - コメントアウト
        # client = AzureOpenAI(
        #     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        #     api_version=os.getenv("OPENAI_API_VERSION"),
        #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        # )
        # model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        # 家用 (ChatGPT)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY が設定されていません")
        
        client = OpenAI(api_key=api_key)
        model_name = "gpt-4o"  # または "gpt-4-vision-preview"
        
        # 改善されたプロンプトの構築
        system_prompt = """
        あなたは画像解析の専門家です。与えられた画像の中から指定された要素を探し、
        その要素が含まれる矩形領域の座標を特定してください。
        
        重要な指示：
        1. 必ず画像を詳しく分析してください
        2. 指定された要素が見つからない場合でも、最も類似している要素を探してください
        3. UI要素、ボタン、テキスト、アイコンなど、あらゆる要素を対象とします
        4. 要素が複数ある場合は、最も目立つものまたは最初に見つかったものを選択してください
        
        座標は以下の形式で必ず返してください：
        x1,y1,x2,y2
        
        ここで：
        - x1, y1: 矩形の左上の座標
        - x2, y2: 矩形の右下の座標
        - 座標は画像の左上を(0,0)とするピクセル単位
        
        例：100,50,200,100
        
        数値のみを返し、他の説明や記号は含めないでください。
        「申し訳ありません」などの謝罪は不要です。必ず数値で座標を返してください。
        """
        
        user_prompt = f"""
        {prompt}の座標を特定してください
        その矩形領域の座標を x1,y1,x2,y2 の形式で返してください。
        
        """
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0,
            max_tokens=100
        )
        
        if not response.choices or not response.choices[0].message.content.strip():
            raise ValueError("LLMから空の応答を受信しました")
        
        # 応答から座標を解析
        coords_text = response.choices[0].message.content.strip()
        print(f"LLMの応答: {coords_text}")
        
        coords = parse_coordinates(coords_text)
        return coords
        
    except Exception as e:
        print(f"LLM呼び出しエラー: {e}")
        return None

def parse_coordinates(coords_text):
    """
    LLMの応答から座標を解析する（改善版）
    
    Args:
        coords_text (str): LLMからの座標文字列
        
    Returns:
        dict: 座標辞書 {'x1': float, 'y1': float, 'x2': float, 'y2': float}
    """
    try:
        # 複数行の場合は最初の行を使用
        first_line = coords_text.split('\n')[0].strip()
        
        # 数字以外の文字を除去してカンマ区切りの数値のみを抽出
        import re
        numbers = re.findall(r'-?\d+\.?\d*', first_line)
        
        if len(numbers) >= 4:
            x1, y1, x2, y2 = map(float, numbers[:4])
        else:
            # 数値が4つ未満の場合は、デフォルト値を使用
            print(f"警告: 座標が不完全です。デフォルト値を使用します。")
            return {
                'x1': 100.0,
                'y1': 100.0,
                'x2': 200.0,
                'y2': 200.0
            }
        
        # 座標の整合性チェックと修正
        if x1 >= x2:
            x1, x2 = x2, x1
        if y1 >= y2:
            y1, y2 = y2, y1
        
        # 負の値を0に修正
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = max(x1 + 10, x2)  # 最小幅を保証
        y2 = max(y1 + 10, y2)  # 最小高さを保証
        
        return {
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2
        }
        
    except Exception as e:
        print(f"座標解析エラー: {e}")
        # エラーの場合はデフォルト値を返す
        print("デフォルト座標を使用します")
        return {
            'x1': 100.0,
            'y1': 100.0,
            'x2': 200.0,
            'y2': 200.0
        }

def scale_coordinates(coords, scale_factor):
    """
    座標を指定倍率でスケーリングする
    
    Args:
        coords (dict): 座標辞書
        scale_factor (float): スケール倍率
        
    Returns:
        dict: スケーリング後の座標辞書
    """
    return {
        'x1': coords['x1'] * scale_factor,
        'y1': coords['y1'] * scale_factor,
        'x2': coords['x2'] * scale_factor,
        'y2': coords['y2'] * scale_factor
    }

def expand_rectangle(coords, expansion_factor):
    """
    矩形を中心から指定倍率で拡張する
    
    Args:
        coords (dict): 座標辞書
        expansion_factor (float): 拡張倍率
        
    Returns:
        dict: 拡張後の座標辞書
    """
    # 現在の矩形の中心を計算
    center_x = (coords['x1'] + coords['x2']) / 2
    center_y = (coords['y1'] + coords['y2']) / 2
    
    # 現在の幅と高さを計算
    width = coords['x2'] - coords['x1']
    height = coords['y2'] - coords['y1']
    
    # 新しい幅と高さを計算
    new_width = width * expansion_factor
    new_height = height * expansion_factor
    
    # 新しい座標を計算
    new_x1 = center_x - new_width / 2
    new_y1 = center_y - new_height / 2
    new_x2 = center_x + new_width / 2
    new_y2 = center_y + new_height / 2
    
    # 座標が負の値にならないように調整
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    
    return {
        'x1': new_x1,
        'y1': new_y1,
        'x2': new_x2,
        'y2': new_y2
    }

# テスト用の関数
if __name__ == "__main__":
    # テスト実行
    test_prompt = "ボタン"
    test_image_path = "test_screenshot.png"
    test_storage_name = "test_rectangle"
    
    findtextfromwholess_consideringresizingimage_storagenameofrectangle(
        test_prompt, 
        test_image_path, 
        test_storage_name
    )
