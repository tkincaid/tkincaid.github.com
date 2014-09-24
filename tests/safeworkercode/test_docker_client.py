import unittest
from mock import patch
from safeworkercode.docker_client import DockerClient

class TestDockerClient(unittest.TestCase):

    def setUp(self):
        self.docker_client = DockerClient()

    images = [
        {
            u'Created': 1392152237,
            u'Id': u'b2fd142e8baf1e28fde73ab2199756050e87143dc52b7ffe58f8a212332628f3',
            u'ParentId': u'6eec79cf6985e1aab65237cdc5e6e87828d4fb39c709c19edb1487519220534d',
            u'RepoTags': [u'ide-52dc081379cbafddb41ca40e-52fa5358637aba0149a9d1ea:latest'],
            u'Size': 405854,
            u'VirtualSize': 1513071025
        },
        {
            u'Created': 1391448149,
            u'Id': u'680031b123088c083964a1a0a0ee016bbe4a44832028ef48d81a6ba4c8d8f022',
            u'ParentId': u'6eec79cf6985e1aab65237cdc5e6e87828d4fb39c709c19edb1487519220534d',
            u'RepoTags': [u'ide-52dc081379cbafddb41ca40e-52fa5358637aba0149a9d1ea:latest'],
            u'Size': 204371253,
            u'VirtualSize': 204371253
        }
    ]

    containers = [
        {
            u'Command': u'python /home/isolated_user/workspace/isolated_user_worker.py',
            u'Created': 1392175644,
            u'Id': u'da3c597229d3fe8dbb5d781a39417ce3a1979418d66f82045b91fadf7fe5e24d',
            u'Image': u'52faea17637aba0d3098eddb-22:latest',
            u'Names': [u'/dreamy_heisenberg'],
            u'Ports': [],
            u'Status': u'Exit 0'},
            {u'Command': u'python /home/isolated_user/workspace/isolated_user_worker.py',
            u'Created': 1392175638,
            u'Id': u'bd5aab138c4e1f9f7043a6dcfc03d9c0417682704ee563db2797e57765828e02',
            u'Image': u'52faea0f637aba0bc198eddb-22:latest',
            u'Names': [u'/elegant_mccarthy'],
            u'Ports': [],
            u'Status': u'Exit 0'
        }
    ]

    def test_find_image_by_id_returns_image_if_found(self):
      with patch.object(self.docker_client, 'images') as mock_docker_client:
        mock_docker_client.return_value = self.images

        image_id = 'b2fd142e8baf1e28fde73ab2199756050e87143dc52b7ffe58f8a212332628f3'

        image = self.docker_client.find_image_by_id(image_id)

        self.assertIsNotNone(image)

    def test_find_image_by_id_returns_none_if_not_found(self):
      with patch.object(self.docker_client, 'images') as mock_docker_client:
        mock_docker_client.return_value = self.images
        image_id = 'x'

        image = self.docker_client.find_image_by_id(image_id)

        self.assertIsNone(image)

    def test_find_image_by_name_returns_image_if_found(self):
      with patch.object(self.docker_client, 'images') as mock_docker_client:
        mock_docker_client.return_value = self.images

        image_id = 'ide-52dc081379cbafddb41ca40e-52fa5358637aba0149a9d1ea'

        image = self.docker_client.find_image_by_name(image_id)

        self.assertIsNotNone(image)

    def test_find_image_by_name_returns_none_if_not_found(self):
      with patch.object(self.docker_client, 'images') as mock_docker_client:
        mock_docker_client.return_value = self.images
        image_id = 'x'

        image = self.docker_client.find_image_by_name(image_id)

        self.assertIsNone(image)

    def test_find_container(self):
        with patch.object(self.docker_client, 'containers') as mock_docker_client:
            mock_docker_client.return_value = self.containers
            expected_container = self.containers[0]

            actual_container = self.docker_client.find_container(expected_container)

            self.assertEqual(expected_container, actual_container)

    def test_get_container_names(self):
        with patch.object(self.docker_client, 'containers') as mock_docker_client:
            mock_docker_client.return_value = self.containers
            expected = self.containers[0]['Names'] + self.containers[1]['Names']
            self.assertEqual(expected, self.docker_client.get_container_names(show_all=True))
            mock_docker_client.return_value = []
            self.assertEqual([], self.docker_client.get_container_names())
