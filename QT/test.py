import numpy as np

class ClusterProcessor:
    def process_array_clusters(self, row):
        """
        주어진 행에서 클러스터를 식별하여 리스트로 반환합니다.
        각 클러스터는 딕셔너리의 목록으로 표현됩니다.
        """
        clusters = []
        current_cluster = []
        zero_count = 0

        for index, value in enumerate(row):
            if value != 0:
                current_cluster.append({'row': None, 'index': index, 'value': value})
                zero_count = 0
            else:
                zero_count += 1
                if zero_count == 2:
                    if current_cluster:
                        clusters.append(current_cluster)
                        current_cluster = []
                    zero_count = 0

        if current_cluster:
            clusters.append(current_cluster)

        return clusters

    def summarize_clusters(self, clusters, row_index):
        """
        클러스터 목록을 받아 각 클러스터에 대한 요약 정보를 반환합니다.
        """
        summaries = []

        for cluster in clusters:
            if not cluster:
                continue

            for item in cluster:
                item['row'] = row_index

            indices = [item['index'] for item in cluster]
            values = [item['value'] for item in cluster if item['value'] != 0]

            if not values:
                continue

            index_mean = round(sum(indices) / len(indices))
            value_mean = round(sum(values) / len(values))

            summary = {
                'row': row_index,
                'index_mean': index_mean,
                'value_mean': value_mean
            }

            summaries.append(summary)

        return summaries

    def process_and_summarize_all(self, data):
        """
        주어진 데이터의 각 행에 대해 클러스터를 식별하고 요약 정보를 생성하여 반환합니다.
        
        :param data: 2차원 리스트 형태의 정수 데이터
        :return: 클러스터 요약 정보가 담긴 리스트
        """
        all_summaries = []

        for row_index, row in enumerate(data):
            clusters = self.process_array_clusters(row)
            summaries = self.summarize_clusters(clusters, row_index)
            all_summaries.extend(summaries)

        return all_summaries

# Example usage
data = [
    np.array([0, 0, 0, 0, 1426, 1388, 1343, 0, 0]),
    np.array([0, 1598, 1599, 0, 0, 1354, 1671, 0, 0]),
    np.array([1400, 1391, 0, 0, 0, 1382, 0, 1364, 1367, 1396])
]

processor = ClusterProcessor()
all_summaries = processor.process_and_summarize_all(data)

# 결과 출력
for summary in all_summaries:
    print(summary)