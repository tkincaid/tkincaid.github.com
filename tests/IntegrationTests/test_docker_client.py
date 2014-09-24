import unittest
from random import randint
import os

from safeworkercode.docker_client import DockerClient
import docker
import time

class TestDockerClient(unittest.TestCase):

    BUSY_LXC_COMMAND = '/bin/sleep 10'

    def setUp(self):
        self.docker_client = DockerClient()
        self.docker_client.pull('busybox')

    def test_remove_container_on_existing_container(self):
        original_container = self.docker_client.create_container('busybox', self.BUSY_LXC_COMMAND, name="test_container")
        self.docker_client.start(original_container)

        timeout = time.time() + 3

        while time.time() < timeout:
            container = self.docker_client.find_container(original_container)
            if container:
                break

        self.assertIsNotNone(container, 'Could not find: {0}'.format(original_container))

        self.assertEqual(self.docker_client.find_container_by_name("test_container")['Id'], container['Id'])

        self.docker_client.kill(container)
        self.docker_client.remove_container(container)

        container = self.docker_client.find_container(container)
        self.assertIsNone(container, 'Could not remove: {}'.format(container))


    def test_remove_container_fails_quietely(self):
        container = 'does-not-exist'
        self.assertRaises(docker.errors.APIError, self.docker_client.remove_container, container)

        self.docker_client.remove_container(container, quiet = True)

    def test_remove_container_and_image(self):

        name = 'test_image_name_{}'.format(randint(1,1000))

        image = self.docker_client.find_image_by_name(name)
        self.assertIsNone(image)

        container = self.docker_client.create_container('busybox', self.BUSY_LXC_COMMAND)

        self.docker_client.commit(container['Id'], name)

        container2 = self.docker_client.create_container(name, self.BUSY_LXC_COMMAND)

        self.docker_client.start(container2)

        image = self.docker_client.find_image_by_name(name)
        self.assertIsNotNone(image, 'Failed to find image from name: ' +  name)

        self.docker_client.remove_container_and_image(container2)

        image = self.docker_client.find_image_by_name(name)
        self.assertIsNone(image)

    def test_copy_files_to_volume(self):
        mount_dest = '/mnt'
        mount_origin = os.path.join(os.path.dirname(__file__),'../testdata')

        filename = 'shared.txt'
        shared_file = os.path.join(mount_origin, filename)

        with open(shared_file, 'w'):
            container = self.docker_client.create_container(
                'busybox',
                ['ls', mount_dest], volumes=[mount_dest]
            )

            self.docker_client.start(container, binds={mount_origin: mount_dest})
            exitcode = self.docker_client.wait(container)
            self.assertEqual(exitcode, 0)
            logs = self.docker_client.logs(container)
            self.assertIn(filename, logs)

        os.unlink(shared_file)

if __name__ == '__main__':
    unittest.main()
