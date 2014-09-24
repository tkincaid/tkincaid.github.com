
import unittest
import psutil
from mock import patch
import urllib2
import pytest

from config.engine import EngConfig
from ModelingMachine.engine.workrequest import mongodict
from ModelingMachine.engine.awsutils import WorkerAWSUtils
import ModelingMachine.engine.awsutils as awsutils

class fakeURL():
    def __init__(self,arg):
        self.arg = arg

    def raise_for_status(self):
        return

    @property
    def text(self):
        if 'amazon' in self.arg:
            return '''
  {
    "vers": 0.01,
    "config": {
        "rate": "perhr",
        "valueColumns": [
            "linux"
        ],
        "currencies": [
            "USD"
        ],
        "regions": [
            {
                "region": "us-east",
                "instanceTypes": [
                    {
                        "type": "generalCurrentGen",
                        "sizes": [
                            {
                                "size": "m3.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.450"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m3.2xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.900"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "generalPreviousGen",
                        "sizes": [
                            {
                                "size": "m1.small",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.060"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m1.medium",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.120"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m1.large",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.240"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m1.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.480"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "computeCurrentGen",
                        "sizes": [
                            {
                                "size": "c3.large",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.150"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "c3.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.300"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "c3.2xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.600"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "c3.4xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "1.200"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "c3.8xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "2.400"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "computePreviousGen",
                        "sizes": [
                            {
                                "size": "c1.medium",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.145"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "c1.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.580"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "cc2.8xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "2.400"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "gpuCurrentGen",
                        "sizes": [
                            {
                                "size": "g2.2xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.650"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "gpuPreviousGen",
                        "sizes": [
                            {
                                "size": "cg1.4xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "2.100"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "hiMemCurrentGen",
                        "sizes": [
                            {
                                "size": "m2.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.410"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m2.2xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.820"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m2.4xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "1.640"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "cr1.8xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "3.500"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "storageCurrentGen",
                        "sizes": [
                            {
                                "size": "hi1.4xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "3.100"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "hs1.8xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "4.600"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "uI",
                        "sizes": [
                            {
                                "size": "t1.micro",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.020"
                                        }
                                    }
                                ]
                            }
                        ]
                    }
                ]
            },
            {
                "region": "us-west-2",
                "instanceTypes": [
                    {
                        "type": "generalCurrentGen",
                        "sizes": [
                            {
                                "size": "m3.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.450"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m3.2xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.900"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "generalPreviousGen",
                        "sizes": [
                            {
                                "size": "m1.small",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.060"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m1.medium",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.120"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m1.large",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.240"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m1.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.480"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "computeCurrentGen",
                        "sizes": [
                            {
                                "size": "c3.large",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.150"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "c3.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.300"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "c3.2xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.600"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "c3.4xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "1.200"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "c3.8xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "2.400"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "computePreviousGen",
                        "sizes": [
                            {
                                "size": "c1.medium",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.145"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "c1.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.580"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "cc2.8xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "2.400"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "gpuCurrentGen",
                        "sizes": [
                            {
                                "size": "g2.2xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.650"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "hiMemCurrentGen",
                        "sizes": [
                            {
                                "size": "m2.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.410"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m2.2xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.820"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m2.4xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "1.640"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "cr1.8xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "3.500"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "storageCurrentGen",
                        "sizes": [
                            {
                                "size": "hi1.4xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "3.100"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "hs1.8xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "4.600"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "uI",
                        "sizes": [
                            {
                                "size": "t1.micro",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.020"
                                        }
                                    }
                                ]
                            }
                        ]
                    }
                ]
            },
            {
                "region": "us-west",
                "instanceTypes": [
                    {
                        "type": "generalCurrentGen",
                        "sizes": [
                            {
                                "size": "m3.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.495"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m3.2xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.990"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "generalPreviousGen",
                        "sizes": [
                            {
                                "size": "m1.small",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.065"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m1.medium",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.130"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m1.large",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.260"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m1.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.520"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "computePreviousGen",
                        "sizes": [
                            {
                                "size": "c1.medium",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.165"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "c1.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.660"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "gpuCurrentGen",
                        "sizes": [
                            {
                                "size": "g2.2xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.702"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "hiMemCurrentGen",
                        "sizes": [
                            {
                                "size": "m2.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.460"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m2.2xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.920"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m2.4xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "1.840"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "uI",
                        "sizes": [
                            {
                                "size": "t1.micro",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.025"
                                        }
                                    }
                                ]
                            }
                        ]
                    }
                ]
            },
            {
                "region": "eu-ireland",
                "instanceTypes": [
                    {
                        "type": "generalCurrentGen",
                        "sizes": [
                            {
                                "size": "m3.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.495"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m3.2xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.990"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "generalPreviousGen",
                        "sizes": [
                            {
                                "size": "m1.small",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.065"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m1.medium",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.130"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m1.large",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.260"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m1.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.520"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "computeCurrentGen",
                        "sizes": [
                            {
                                "size": "c3.large",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.171"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "c3.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.342"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "c3.2xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.683"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "c3.4xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "1.366"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "c3.8xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "2.732"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "computePreviousGen",
                        "sizes": [
                            {
                                "size": "c1.medium",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.165"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "c1.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.660"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "cc2.8xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "2.700"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "gpuCurrentGen",
                        "sizes": [
                            {
                                "size": "g2.2xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.702"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "gpuPreviousGen",
                        "sizes": [
                            {
                                "size": "cg1.4xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "2.36"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "hiMemCurrentGen",
                        "sizes": [
                            {
                                "size": "m2.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.460"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m2.2xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.920"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m2.4xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "1.840"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "cr1.8xlarge",
                                "valueColumns": [
                                    {
                                       "name": "linux",
                                        "prices": {
                                            "USD": "3.750"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "storageCurrentGen",
                        "sizes": [
                            {
                                "size": "hi1.4xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "3.410"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "hs1.8xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "4.900"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "uI",
                        "sizes": [
                            {
                                "size": "t1.micro",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.020"
                                        }
                                    }
                                ]
                            }
                        ]
                    }
                ]
            },
            {
                "region": "apac-sin",
                "instanceTypes": [
                    {
                        "type": "generalCurrentGen",
                        "sizes": [
                            {
                                "size": "m3.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.630"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m3.2xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "1.260"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "generalPreviousGen",
                        "sizes": [
                            {
                                "size": "m1.small",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.080"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m1.medium",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.160"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m1.large",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.320"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m1.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.640"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "computeCurrentGen",
                        "sizes": [
                            {
                                "size": "c3.large",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.189"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "c3.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.378"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "c3.2xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.756"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "c3.4xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "1.512"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "c3.8xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "3.024"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "computePreviousGen",
                        "sizes": [
                            {
                                "size": "c1.medium",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.183"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "c1.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.730"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "hiMemCurrentGen",
                        "sizes": [
                            {
                                "size": "m2.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.495"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m2.2xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.990"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m2.4xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "1.980"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "storageCurrentGen",
                        "sizes": [
                            {
                                "size": "hs1.8xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "5.570"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "uI",
                        "sizes": [
                            {
                                "size": "t1.micro",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.020"
                                        }
                                    }
                                ]
                            }
                        ]
                    }
                ]
            },
            {
                "region": "apac-tokyo",
                "instanceTypes": [
                    {
                        "type": "generalCurrentGen",
                        "sizes": [
                            {
                                "size": "m3.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.684"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m3.2xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "1.368"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "generalPreviousGen",
                        "sizes": [
                            {
                                "size": "m1.small",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.088"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m1.medium",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.175"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m1.large",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.350"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m1.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.700"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "computeCurrentGen",
                        "sizes": [
                            {
                                "size": "c3.large",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.192"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "c3.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.383"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "c3.2xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.766"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "c3.4xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "1.532"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "c3.8xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "3.064"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "computePreviousGen",
                        "sizes": [
                            {
                                "size": "c1.medium",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.185"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "c1.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.740"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "cc2.8xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "2.960"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "hiMemCurrentGen",
                        "sizes": [
                            {
                                "size": "m2.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.505"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m2.2xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "1.010"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m2.4xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "2.020"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "cr1.8xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "4.310"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "storageCurrentGen",
                        "sizes": [
                            {
                                "size": "hi1.4xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "3.820"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "hs1.8xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "5.670"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "uI",
                        "sizes": [
                            {
                                "size": "t1.micro",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.027"
                                        }
                                    }
                                ]
                            }
                        ]
                    }
                ]
            },
            {
                "region": "apac-syd",
                "instanceTypes": [
                    {
                        "type": "generalCurrentGen",
                        "sizes": [
                            {
                                "size": "m3.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.630"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m3.2xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "1.260"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "generalPreviousGen",
                        "sizes": [
                            {
                                "size": "m1.small",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.080"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m1.medium",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.160"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m1.large",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.320"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m1.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.640"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "computeCurrentGen",
                        "sizes": [
                            {
                                "size": "c3.large",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.189"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "c3.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.378"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "c3.2xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.756"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "c3.4xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "1.512"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "c3.8xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "3.024"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "computePreviousGen",
                        "sizes": [
                            {
                                "size": "c1.medium",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.183"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "c1.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.730"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "hiMemCurrentGen",
                        "sizes": [
                            {
                                "size": "m2.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.495"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m2.2xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.990"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m2.4xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "1.980"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "storageCurrentGen",
                        "sizes": [
                            {
                                "size": "hs1.8xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "5.570"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "uI",
                        "sizes": [
                            {
                                "size": "t1.micro",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.020"
                                        }
                                    }
                                ]
                            }
                        ]
                    }
                ]
            },
            {
                "region": "sa-east-1",
                "instanceTypes": [
                    {
                        "type": "generalCurrentGen",
                        "sizes": [
                            {
                                "size": "m3.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.612"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m3.2xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "1.224"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "generalPreviousGen",
                        "sizes": [
                            {
                                "size": "m1.small",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.080"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m1.medium",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.160"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m1.large",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.320"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m1.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.640"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "computePreviousGen",
                        "sizes": [
                            {
                                "size": "c1.medium",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.200"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "c1.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.800"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "hiMemCurrentGen",
                        "sizes": [
                            {
                                "size": "m2.xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.540"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m2.2xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "1.080"
                                        }
                                    }
                                ]
                            },
                            {
                                "size": "m2.4xlarge",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "2.160"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "uI",
                        "sizes": [
                            {
                                "size": "t1.micro",
                                "valueColumns": [
                                    {
                                        "name": "linux",
                                        "prices": {
                                            "USD": "0.027"
                                        }
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    }
}
'''
        else:
            return 'us-east-1a'

class TestWorkerAWSUtils(unittest.TestCase):

    def test_update_instance_type(self):
        w = WorkerAWSUtils(None)
        for i in range(2):
            out = w.update_instance_type(i,None)
            answer = {'ec2.reps='+str(i):
                        {'spot_price': 0, 'on_demand_price': 0, 'availability_zone': 'None',
                            'instance_size': 'None', 'CPU_count': psutil.NUM_CPUS, 'instance_type': 'local',
                            'workers': EngConfig['SECURE_WORKER_FIT_WORKERS']
                        }
                    }
            self.assertEqual(mongodict(out), mongodict(answer))

    @pytest.mark.skip
    def test_update_instance_type_AWS(self):
        w = WorkerAWSUtils('any')
        with patch('ModelingMachine.engine.awsutils.urllib2.urlopen') as mock_url:
            mock_url.side_effect = fakeURL
            for i in range(2):
                out = w.update_instance_type(i)
                self.assertIn('ec2',out)
                self.assertIn('reps='+str(i),out['ec2'])
                self.assertIn('availability_zone',out['ec2']['reps='+str(i)])
                self.assertIn('instance_size',out['ec2']['reps='+str(i)])
                self.assertGreater(float(out['ec2']['reps='+str(i)]['spot_price']),0)
                self.assertGreater(float(out['ec2']['reps='+str(i)]['on_demand_price']),0)
                self.assertGreater(int(out['ec2']['reps='+str(i)]['CPU_count']),0)
                self.assertGreater(int(out['ec2']['reps='+str(i)]['workers']),0)
                self.assertIn(out['ec2']['reps='+str(i)]['instance_type'],('on_demand','spot','reserved'))

    def test_aggregate_task_info(self):
        s_time = 1
        e_time = 3

        w = WorkerAWSUtils(None)
        report = w.update_instance_type(1)

        parts = {'max_reps':1}
        part_reps = max([i[0]+1 for i in parts.get('partitions',[[parts.get('max_reps',0)-1,-1]])])
        timer_key = 'reps='+str(part_reps)
        timer_values = {
                'finish_time':     {timer_key: e_time},
                'time': {'start_time': {timer_key: s_time},
                         'finish_time':{timer_key: e_time},
                         'total_time': {timer_key: (e_time - s_time)}
                         }
                }
        report.update(timer_values)

        report['task_info'] = {u'reps=1': [
            [{u'fit max RAM': 0, u'fit CPU pct': 0, u'fit CPU time': 0.010000000000000675,
                u'transform max RAM': 0, u'fit clock time': 0.014407873153686523, u'fit avg RAM': 0, u'cached': True,
                u'ytrans': None, u'transform avg RAM': 0, u'transform clock time': 0.020978212356567383, u'version': u'0.1',
                u'fit total RAM': 8060366848L, u'task_name': u'NI', u'transform CPU time': 0.020000000000000462,
                u'transform total RAM': 8060366848L, u'transform CPU pct': 0, u'arguments': None}],
            [{u'fit max RAM': 0, u'fit CPU pct': 0, u'fit CPU time': 0.03000000000000025, u'predict CPU pct': 0,
                u'fit clock time': 0.023867130279541016, u'task_name': u'GLMB', u'fit avg RAM': 0,
                u'cached': True, u'ytrans': None, u'fit total RAM': 8060366848L, u'predict CPU time': 0.020000000000000462,
                u'predict total RAM': 8060366848L, u'predict clock time': 0.009172916412353516, u'version': u'0.1',
                u'predict avg RAM': 0, u'arguments': None, u'predict max RAM': 0}]
            ]}

        report['job_info'] = {'reps=1': {}}

        out = w.aggregate_task_info(report)

        self.assertIsInstance(out,dict)
        self.assertEqual(out.keys(),['reps=1'])
        self.assertIsInstance(out['reps=1'],dict)

        keys = ['total_cpu_time', 'cpu_usage', 'max_noncached_ram', 'bp_cost', 'noncached_cost',
                'rate', 'cost', 'total_clock_time', 'noncached_cpu_usage', 'bp_time', 'cached',
                'total_noncached_cpu_time', 'max_ram', 'total_noncached_clock_time']

        version2_keys = ['version', 'total_cpu_time', 'total_clock_time', 'max_ram',
                         'cost', 'cached_cpu_time', 'cached_clock_time', 'max_cached_ram',
                         'cached_cost', 'cpu_usage', 'rate']
        if out['reps=1'].get('version') == 2:
            self.assertEqual(set(out['reps=1'].keys()),set(version2_keys))
        else:
            self.assertEqual(set(out['reps=1'].keys()),set(keys))


    def test_timer_values(self):
        s_time = 1
        e_time = 3

        parts = {'max_reps':1}
        part_reps = max([i[0]+1 for i in parts.get('partitions',[[parts.get('max_reps',0)-1,-1]])])

        old_timer_values = {
                'finish_time.reps='+str(parts.get('max_reps',part_reps)):e_time,
                'time.start_time.reps='+str(parts.get('max_reps',part_reps)):s_time,
                'time.finish_time.reps='+str(parts.get('max_reps',part_reps)):e_time,
                'time.total_time.reps='+str(parts.get('max_reps',part_reps)):(e_time - s_time)
                }

        timer_key = 'reps='+str(part_reps)
        timer_values = {
                'finish_time':     {timer_key: e_time},
                'time': {'start_time': {timer_key: s_time},
                         'finish_time':{timer_key: e_time},
                         'total_time': {timer_key: (e_time - s_time)}
                         }
                }

        self.assertEqual(mongodict(timer_values), old_timer_values)

    def test_get_instance_info(self):
        with patch('ModelingMachine.engine.awsutils.WorkerAWSUtils.get_current_zone') as mock_zone:
            mock_zone.return_value = 'us-east-1b'
            w = WorkerAWSUtils(None)
            info = w.get_instance_info(None)
            self.assertEqual(info['instance_type'], 'local')

    def test_get_ond_price(self):
        with patch('ModelingMachine.engine.awsutils.requests.get') as mock_url:
            mock_url.side_effect = fakeURL
            w = WorkerAWSUtils(None)
            region = 'us-east'
            size = 'm1.medium'
            price = w.get_ond_price(region,size)
            self.assertGreater(float(price),0)

if __name__ == '__main__':
    unittest.main()
