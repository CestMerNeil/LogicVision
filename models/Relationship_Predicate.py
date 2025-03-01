import ltn
import torch
import torch.nn as nn


class In(ltn.Predicate):
    """Neural network predicate for the 'In' relationship.

    This predicate uses a feed-forward neural network to compute the likelihood
    of the 'In' relationship between a subject and an object.

    Attributes:
        net (torch.nn.Sequential): The underlying neural network model.
    """

    def __init__(self, input_dim):
        """Initializes the 'In' predicate network.

        Args:
            input_dim (int): The dimension of the input feature vector for each object.
        """
        net = nn.Sequential(
            nn.Linear(input_dim * 2, 1024),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        super().__init__(model=net)
        self.net = net

    def forward(self, subj: ltn.Variable, obj: ltn.Variable) -> torch.Tensor:
        """Computes the 'In' predicate score for the given subject and object.

        The subject and object feature tensors are concatenated and reshaped before
        being passed through the network.

        Args:
            subj (ltn.Variable): The subject variable containing feature values.
            obj (ltn.Variable): The object variable containing feature values.

        Returns:
            torch.Tensor: The computed predicate score as a probability.
        """
        device = next(self.net.parameters()).device
        subj_tensor = subj.value.to(device)
        obj_tensor = obj.value.to(device)
        subj_tensor = subj_tensor.unsqueeze(1)
        obj_tensor = obj_tensor.unsqueeze(1)
        obj_tensor = obj_tensor.expand_as(subj_tensor)
        combined = torch.cat([subj_tensor, obj_tensor], dim=-1).view(-1, 10)
        return self.net(combined)


class On(ltn.Predicate):
    """Neural network predicate for the 'On' relationship.

    This predicate uses a feed-forward neural network to compute the likelihood
    of the 'On' relationship between a subject and an object.

    Attributes:
        net (torch.nn.Sequential): The underlying neural network model.
    """

    def __init__(self, input_dim):
        """Initializes the 'On' predicate network.

        Args:
            input_dim (int): The dimension of the input feature vector for each object.
        """
        net = nn.Sequential(
            nn.Linear(input_dim * 2, 1024),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        super().__init__(model=net)
        self.net = net

    def forward(self, subj: ltn.Variable, obj: ltn.Variable) -> torch.Tensor:
        """Computes the 'On' predicate score for the given subject and object.

        The subject and object feature tensors are concatenated and reshaped before
        being passed through the network.

        Args:
            subj (ltn.Variable): The subject variable containing feature values.
            obj (ltn.Variable): The object variable containing feature values.

        Returns:
            torch.Tensor: The computed predicate score as a probability.
        """
        device = next(self.net.parameters()).device
        subj_tensor = subj.value.to(device)
        obj_tensor = obj.value.to(device)
        subj_tensor = subj_tensor.unsqueeze(1)
        obj_tensor = obj_tensor.unsqueeze(1)
        obj_tensor = obj_tensor.expand_as(subj_tensor)
        combined = torch.cat([subj_tensor, obj_tensor], dim=-1).view(-1, 10)
        return self.net(combined)


class NextTo(ltn.Predicate):
    """Neural network predicate for the 'NextTo' relationship.

    This predicate uses a feed-forward neural network to compute the likelihood
    of the 'NextTo' relationship between a subject and an object.

    Attributes:
        net (torch.nn.Sequential): The underlying neural network model.
    """

    def __init__(self, input_dim):
        """Initializes the 'NextTo' predicate network.

        Args:
            input_dim (int): The dimension of the input feature vector for each object.
        """
        net = nn.Sequential(
            nn.Linear(input_dim * 2, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        super().__init__(model=net)
        self.net = net

    def forward(self, subj: ltn.Variable, obj: ltn.Variable) -> torch.Tensor:
        """Computes the 'NextTo' predicate score for the given subject and object.

        The subject and object feature tensors are concatenated and reshaped before
        being passed through the network.

        Args:
            subj (ltn.Variable): The subject variable containing feature values.
            obj (ltn.Variable): The object variable containing feature values.

        Returns:
            torch.Tensor: The computed predicate score as a probability.
        """
        device = next(self.net.parameters()).device
        subj_tensor = subj.value.to(device)
        obj_tensor = obj.value.to(device)
        subj_tensor = subj_tensor.unsqueeze(1)
        obj_tensor = obj_tensor.unsqueeze(1)
        obj_tensor = obj_tensor.expand_as(subj_tensor)
        combined = torch.cat([subj_tensor, obj_tensor], dim=-1).view(-1, 10)
        return self.net(combined)


class OnTopOf(ltn.Predicate):
    """Neural network predicate for the 'OnTopOf' relationship.

    This predicate uses a feed-forward neural network to compute the likelihood
    of the 'OnTopOf' relationship between a subject and an object.

    Attributes:
        net (torch.nn.Sequential): The underlying neural network model.
    """

    def __init__(self, input_dim):
        """Initializes the 'OnTopOf' predicate network.

        Args:
            input_dim (int): The dimension of the input feature vector for each object.
        """
        net = nn.Sequential(
            nn.Linear(input_dim * 2, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        super().__init__(model=net)
        self.net = net

    def forward(self, subj: ltn.Variable, obj: ltn.Variable) -> torch.Tensor:
        """Computes the 'OnTopOf' predicate score for the given subject and object.

        The subject and object feature tensors are concatenated and reshaped before
        being passed through the network.

        Args:
            subj (ltn.Variable): The subject variable containing feature values.
            obj (ltn.Variable): The object variable containing feature values.

        Returns:
            torch.Tensor: The computed predicate score as a probability.
        """
        device = next(self.net.parameters()).device
        subj_tensor = subj.value.to(device)
        obj_tensor = obj.value.to(device)
        subj_tensor = subj_tensor.unsqueeze(1)
        obj_tensor = obj_tensor.unsqueeze(1)
        obj_tensor = obj_tensor.expand_as(subj_tensor)
        combined = torch.cat([subj_tensor, obj_tensor], dim=-1).view(-1, 10)
        return self.net(combined)


class Near(ltn.Predicate):
    """Neural network predicate for the 'Near' relationship.

    This predicate uses a feed-forward neural network to compute the likelihood
    of the 'Near' relationship between a subject and an object.

    Attributes:
        net (torch.nn.Sequential): The underlying neural network model.
    """

    def __init__(self, input_dim):
        """Initializes the 'Near' predicate network.

        Args:
            input_dim (int): The dimension of the input feature vector for each object.
        """
        net = nn.Sequential(
            nn.Linear(input_dim * 2, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        super().__init__(model=net)
        self.net = net

    def forward(self, subj: ltn.Variable, obj: ltn.Variable) -> torch.Tensor:
        """Computes the 'Near' predicate score for the given subject and object.

        The subject and object feature tensors are concatenated and reshaped before
        being passed through the network.

        Args:
            subj (ltn.Variable): The subject variable containing feature values.
            obj (ltn.Variable): The object variable containing feature values.

        Returns:
            torch.Tensor: The computed predicate score as a probability.
        """
        device = next(self.net.parameters()).device
        subj_tensor = subj.value.to(device)
        obj_tensor = obj.value.to(device)
        subj_tensor = subj_tensor.unsqueeze(1)
        obj_tensor = obj_tensor.unsqueeze(1)
        obj_tensor = obj_tensor.expand_as(subj_tensor)
        combined = torch.cat([subj_tensor, obj_tensor], dim=-1).view(-1, 10)
        return self.net(combined)


class Under(ltn.Predicate):
    """Neural network predicate for the 'Under' relationship.

    This predicate uses a feed-forward neural network to compute the likelihood
    of the 'Under' relationship between a subject and an object.

    Attributes:
        net (torch.nn.Sequential): The underlying neural network model.
    """

    def __init__(self, input_dim):
        """Initializes the 'Under' predicate network.

        Args:
            input_dim (int): The dimension of the input feature vector for each object.
        """
        net = nn.Sequential(
            nn.Linear(input_dim * 2, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        super().__init__(model=net)
        self.net = net

    def forward(self, subj: ltn.Variable, obj: ltn.Variable) -> torch.Tensor:
        """Computes the 'Under' predicate score for the given subject and object.

        The subject and object feature tensors are concatenated and reshaped before
        being passed through the network.

        Args:
            subj (ltn.Variable): The subject variable containing feature values.
            obj (ltn.Variable): The object variable containing feature values.

        Returns:
            torch.Tensor: The computed predicate score as a probability.
        """
        device = next(self.net.parameters()).device
        subj_tensor = subj.value.to(device)
        obj_tensor = obj.value.to(device)
        subj_tensor = subj_tensor.unsqueeze(1)
        obj_tensor = obj_tensor.unsqueeze(1)
        obj_tensor = obj_tensor.expand_as(subj_tensor)
        combined = torch.cat([subj_tensor, obj_tensor], dim=-1).view(-1, 10)
        return self.net(combined)
