"""
Go2 policy loss landscape visualization.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_flat_weights(model):
    """Model weight'lerini tek vektör yap."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_weights(model, flat_weights):
    """Tek vektörden weight'leri geri yükle."""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_weights[offset:offset + numel].view_as(p.data))
        offset += numel


def random_direction(shape, device='cuda'):
    """Rastgele normalize yön."""
    direction = torch.randn(shape, device=device)
    direction = direction / direction.norm()
    return direction


def compute_loss_at_point(model, env, theta_star, direction1, direction2, alpha, beta, n_episodes=5):
    """Belirli noktada policy'nin loss/reward'ını hesapla."""
    # Weight'leri ayarla
    new_weights = theta_star + alpha * direction1 + beta * direction2
    set_flat_weights(model, new_weights)

    # Birkaç episode çalıştır
    total_reward = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        for step in range(1000):
            with torch.no_grad():
                action = model(obs)
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward.item()
            if done.any():
                break
        total_reward += episode_reward

    # Negative reward = loss gibi düşün
    return -total_reward / n_episodes


def visualize_policy_landscape(checkpoint_path, env_name, resolution=21):
    """
    Policy'nin loss landscape'ini 2D ve 3D görselleştir.
    """
    # Model yükle
    checkpoint = torch.load(checkpoint_path)
    model = create_policy_network(...)  # Senin network config'in
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Environment oluştur
    env = gym.make(env_name)

    # Mevcut optimal weight'ler
    theta_star = get_flat_weights(model)

    # Rastgele iki yön
    d1 = random_direction(theta_star.shape)
    d2 = random_direction(theta_star.shape)

    # Grid
    alphas = np.linspace(-0.5, 0.5, resolution)
    betas = np.linspace(-0.5, 0.5, resolution)

    loss_surface = np.zeros((resolution, resolution))

    print("Computing loss landscape...")
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            loss_surface[i, j] = compute_loss_at_point(
                model, env, theta_star, d1, d2, alpha, beta
            )
            print(f"  ({i},{j}): loss = {loss_surface[i, j]:.2f}")

    # Weight'leri geri yükle
    set_flat_weights(model, theta_star)

    # 2D Contour Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Contour
    cs = axes[0].contourf(alphas, betas, loss_surface.T, levels=30, cmap='viridis')
    axes[0].plot(0, 0, 'r*', markersize=15, label='Trained model')
    axes[0].set_xlabel('Direction 1')
    axes[0].set_ylabel('Direction 2')
    axes[0].set_title('2D Loss Landscape (Contour)')
    axes[0].legend()
    plt.colorbar(cs, ax=axes[0], label='Negative Reward')

    # Right: 3D Surface
    ax3d = fig.add_subplot(122, projection='3d')
    A, B = np.meshgrid(alphas, betas)
    ax3d.plot_surface(A, B, loss_surface.T, cmap='viridis', alpha=0.8)
    ax3d.scatter([0], [0], [loss_surface[resolution // 2, resolution // 2]],
                 color='red', s=100, label='Trained model')
    ax3d.set_xlabel('Direction 1')
    ax3d.set_ylabel('Direction 2')
    ax3d.set_zlabel('Negative Reward')
    ax3d.set_title('3D Loss Landscape')

    plt.tight_layout()
    plt.savefig('policy_loss_landscape.png', dpi=150)
    plt.show()

    return loss_surface